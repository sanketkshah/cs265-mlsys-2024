import csv
import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any, Dict, List, Set, cast

import torch
import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

# Minimum memory allocated by PyTorch for a tensor, change according to your device type
_PYTORCH_MIN_ALLOCATE = 512


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3
    RECOMP = 4


class MemStats:
    def __init__(
        self,
        param_and_opt_state_mem: int,
        grad_mem: int,
        act_mem: int,
        other_mem: int,
    ) -> None:
        self.param_and_opt_state_memory = param_and_opt_state_mem
        self.grad_memory = grad_mem
        self.activation_memory = act_mem
        self.other_memory = other_mem


@dataclass
class NodeInfo:
    rank: float = 0.0
    node_type: NodeType | None = None
    run_time: float = 1.0
    start_time: float = 0.0
    swap_time: float = 0.0
    peak_total_mem: int = 0
    mem_stats: MemStats | None = None
    memory_size: int = 0  # size of the tensor in bytes
    cpu_ref: torch.Tensor | None = None
    last_forward_access: fx.Node | None = None
    first_back_access: fx.Node | None = None
    last_forward_uses: List[fx.Node] = field(default_factory=list)
    first_back_uses: List[fx.Node] = field(default_factory=list)
    inactive_time: float = 0.0
    recomp_srcs: Set[fx.Node] = field(default_factory=set)
    recomp_time: float = 0.0
    total_recomp_time: float = 0.0
    recompute_ratio: float = 0.0
    is_recomp: bool = False


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.
class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        self.module = module
        self.node_info: Dict[fx.Node, NodeInfo] = {}
        self.placeholder_nodes: List[fx.Node] = []
        self.intermediate_nodes: List[fx.Node] = []
        self.node_runtimes: Dict[fx.Node, List[float]] = {}
        self.node_swap_times: Dict[fx.Node, List[float]] = {}
        self.forward_end: fx.Node
        self.backward_start: fx.Node
        self.swapped_memory: int = 0
        self.param_and_opt_state_memory: int

        # initial setup and classification of nodes in the computational graph
        self.rank = 0
        for node in self.module.graph.nodes:
            n_info = NodeInfo()
            n_info.rank = self.rank
            self.rank += 1
            # Initially set the node types of all nodes to other
            n_info.node_type = NodeType.OTHER
            self.node_info[node] = n_info
            # Find the forward end and backward start dummy nodes
            if node.name == "sep" and node.target == torch.ops.separator.sep.default:
                self.forward_end = node
            elif (
                node.name == "sep_backward"
                and node.target == torch.ops.separator.sep_backward.default
            ):
                self.backward_start = node
            # Use the optimizer to get the parameter and gradient nodes
            if node.target == torch.ops.aten._fused_adam.default:
                param_adam_args = node.args[0]
                grad_adam_args = node.args[1]

                assert len(param_adam_args) == len(grad_adam_args), (
                    "Unequal number of params and gradients"
                )

                for param in param_adam_args:
                    assert isinstance(param, fx.Node), (
                        "Expected param to be an fx.Node instance"
                    )
                    assert param.op == OP.PLACEHOLDER, (
                        "Expected all params nodes to be of type PLACEHOLDER"
                    )
                    self.node_info[param].node_type = NodeType.PARAM

                for grad in grad_adam_args:
                    assert isinstance(grad, fx.Node), (
                        "Expected grad to be an fx.Node instance"
                    )
                    self.node_info[grad].node_type = NodeType.GRAD

        # identifies and classifies intermediate nodes (activations) in the computational graph
        for node in self.module.graph.nodes:
            if self.node_info[node].rank < self.node_info[self.forward_end].rank:
                if node.op == OP.PLACEHOLDER:
                    self.placeholder_nodes.append(node)
                    continue
                input_nodes: List[fx.Node] = node.all_input_nodes
                input_nodes_op: List[bool] = [
                    self.node_info[n].node_type == NodeType.PARAM for n in input_nodes
                ]
                # if all the input nodes are params, then the node is a param
                if all(input_nodes_op):
                    self.node_info[node].node_type = NodeType.PARAM
                    continue
                users = node.users
                # from the users we get the last forward use
                # and the first backward use using ranks
                last_forward = None
                first_backward = None
                for user in users:
                    u_info = self.node_info[user]
                    if u_info.rank < self.node_info[self.forward_end].rank:
                        if last_forward is None:
                            last_forward = user
                        elif self.node_info[last_forward].rank < u_info.rank:
                            last_forward = user
                    if u_info.rank > self.node_info[self.backward_start].rank:
                        if first_backward is None:
                            first_backward = user
                        elif self.node_info[first_backward].rank > u_info.rank:
                            first_backward = user
                if last_forward is not None and first_backward is not None:
                    n_info = self.node_info[node]
                    self.intermediate_nodes.append(node)
                    n_info.node_type = NodeType.ACT
                    self.node_info[last_forward].last_forward_uses.append(node)
                    self.node_info[first_backward].first_back_uses.append(node)
                    n_info.last_forward_access = last_forward
                    n_info.first_back_access = first_backward
                    print(
                        f"Intermediate Node: {node.name}, Last forward use: {last_forward.name}, First backward use: {first_backward.name}"
                    )

    def _swap_out_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be offloaded
        # 2) Retrieve their CPU reference (if none allocate a CPU tensor in
        #    pinned memory)
        # 3) Copy the tensor to the CPU, add the CPU tensor to the Interpreter
        #    environment
        # 4) Delete the GPU tensor
        nodes_to_offload = self.node_info[node].last_forward_uses
        for o_node in nodes_to_offload:
            o_info = self.node_info[o_node]
            cpu_ref = o_info.cpu_ref
            tensor = self.env[o_node]
            assert isinstance(tensor, torch.Tensor)
            if cpu_ref is None:
                cpu_ref = torch.zeros_like(tensor, device="cpu").pin_memory()
            assert cpu_ref.is_pinned, f"CPU ref is not pinned for {o_node.name}"

            swap_start_event = torch.cuda.Event(enable_timing=True)
            swap_end_event = torch.cuda.Event(enable_timing=True)

            swap_start_event.record()
            cpu_ref = cpu_ref.copy_(tensor, False)
            swap_end_event.record()

            torch.cuda.synchronize()
            o_info.cpu_ref = cpu_ref
            self.env[o_node] = cpu_ref
            del tensor
            tensor = None
            cpu_ref = None
            self.swapped_memory += o_info.memory_size
            swap_time = swap_start_event.elapsed_time(swap_end_event)
            self.node_swap_times.setdefault(o_node, []).append(swap_time)

    def _swap_in_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be prefetched
        # 2) Retrieve their CPU reference (assert if it resides in pinned
        #    memory)
        # 3) Copy the tensor to GPU memory and add it to the Interpreter
        #    environment
        # 4) Update the state of intermediate tensor in NodeInfo
        nodes_to_fetch = self.node_info[node].first_back_uses
        for p_node in nodes_to_fetch:
            # If the node is already in GPU memory, skip it
            if self.env[p_node].is_cuda:
                continue

            # Else, we need to swap it in
            p_info = self.node_info[p_node]
            assert p_info.cpu_ref is not None, f"CPU ref is not set for {p_node.name}"
            assert p_info.cpu_ref.is_pinned, f"CPU ref is not pinned for {p_node.name}"
            cpu_ref = cast(torch.Tensor, p_info.cpu_ref)

            swap_start_event = torch.cuda.Event(enable_timing=True)
            swap_end_event = torch.cuda.Event(enable_timing=True)

            swap_start_event.record()
            tensor = cpu_ref.to(
                device=torch.cuda.current_device(),
                memory_format=torch.preserve_format,
                non_blocking=False,
            )
            swap_end_event.record()
            self.env[p_node] = tensor.contiguous()
            tensor = None
            torch.cuda.synchronize()
            self.swapped_memory -= p_info.memory_size
            assert self.swapped_memory >= 0, (
                f"Swapped memory is less than zero {self.swapped_memory}"
            )
            swap_time = swap_start_event.elapsed_time(swap_end_event)
            self.node_swap_times.setdefault(p_node, []).append(swap_time)

    def _get_recomp_srcs(self, cand, all_possible_recomp_srcs):
        allowed_recomp_srcs = []
        recomp_srcs = cand.all_input_nodes
        recomp_time = self.node_info[cand].run_time
        while len(recomp_srcs) > 0:
            node = recomp_srcs.pop()
            if node in all_possible_recomp_srcs:
                allowed_recomp_srcs.append(node)
            else:
                if len(node.all_input_nodes) > 0:
                    recomp_time += self.node_info[node].run_time
                    recomp_srcs.extend(node.all_input_nodes)
                elif node.name.startswith("arange") or node.name.startswith("ones"):
                    continue  # constant nodes
                else:
                    raise ValueError(f"Node {node.name} has no allowed input nodes")
        return set(allowed_recomp_srcs), recomp_time

    def _initialization(self, candidate_set):
        """
        Initialize each candidate in the candidate set with recomputation sources,
        time estimates, and recomputation ratio according to Algorithm D.
        """
        all_possible_recomp_srcs = self.placeholder_nodes + candidate_set
        for cand in candidate_set:
            # Set recomputation sources as subset of placeholder and other candidates
            self.node_info[cand].recomp_srcs, self.node_info[cand].recomp_time = (
                self._get_recomp_srcs(cand, all_possible_recomp_srcs)
            )
            # Set total recomputation time
            self.node_info[cand].total_recomp_time = self.node_info[
                cand
            ].recomp_time  # recomp_cnt = 1 initially
            # Calculate recomputation ratio
            self.node_info[cand].recompute_ratio = (
                self.node_info[cand].memory_size
                / self.node_info[cand].total_recomp_time
            )

    def _update_recomps(self, cand, recomps):
        """
        Update existing recomputations based on Algorithm H.

        Args:
            cand: Current candidate being processed
            recomps: Set of recomputations to process
        """
        # Initialize recomputation count
        recomp_cnt = 1

        # For each candidate in the recomputation set
        for rp in recomps:
            # If current candidate is in the recomputation sources of rp
            if cand in self.node_info[rp].recomp_srcs:
                # Remove the current candidate from recomputation sources
                self.node_info[rp].recomp_srcs.remove(cand)

                # Add current candidate's recomputation sources
                self.node_info[rp].recomp_srcs.update(self.node_info[cand].recomp_srcs)

                # Update recomputation time
                self.node_info[rp].recomp_time += self.node_info[cand].recomp_time

                # Increment recomputation count
                recomp_cnt += 1

        return recomp_cnt

    def _update_candidates(self, t, recomp_cnt, recomps, candidates):
        """
        Update candidates based on Algorithm I.

        Args:
            t: Target node being processed
            recomp_cnt: Recomputation count
            candidates: Set of candidates to update
        """
        for cand in candidates:
            # If target is in candidate's recomputation sources
            if t in self.node_info[cand].recomp_srcs:
                # Remove target from recomputation sources
                self.node_info[cand].recomp_srcs.remove(t)
                # Add target's recomputation sources
                self.node_info[cand].recomp_srcs.update(self.node_info[t].recomp_srcs)
                # Update recomputation time
                self.node_info[cand].recomp_time += self.node_info[t].recomp_time

                # Set total recomputation time to current recomputation time
                self.node_info[cand].total_recomp_time = self.node_info[
                    cand
                ].recomp_time
                # Account for the cost of recomputing cand if it is a
                # recomputation source for any rp
                for rp in recomps:
                    if cand in self.node_info[rp].recomp_srcs:
                        self.node_info[cand].total_recomp_time += self.node_info[
                            cand
                        ].recomp_time

            # If candidate is in target's recomputation sources
            #   You have to account for the cost of recomputing the target
            #   every time the candidate is recomputed
            elif cand in self.node_info[t].recomp_srcs:
                # Update total recomputation time based on recomputation count
                self.node_info[cand].total_recomp_time = (
                    self.node_info[cand].recomp_time * recomp_cnt
                )

            # Update the recomputation ratio
            self.node_info[cand].recompute_ratio = (
                self.node_info[cand].memory_size
                / self.node_info[cand].total_recomp_time
            )

    def run_checkpoint_selection(self, mem_limit: int):
        # Check that the graph has been profiled
        assert self.is_profiled, (
            "GraphProfiler must be profiled before running checkpoint selection"
        )
        # Input parameters
        candidate_set = self.intermediate_nodes
        max_peak_memory = max(
            self.node_info[node].peak_total_mem for node in self.module.graph.nodes
        )
        # Algorithm B implementation
        mem_consumption = max_peak_memory
        # Initialize candidates
        self._initialization(candidate_set)
        # Initialize empty recomputation set
        recomps = []

        # Main loop
        while len(candidate_set) > 0:
            # Check if memory constraint is satisfied
            if (mem_consumption - mem_limit) <= 0:
                break
            # Get candidate with maximum recomputation benefit
            max_recomp_ratio = max(
                [self.node_info[cand].recompute_ratio for cand in candidate_set]
            )
            r_cand = next(
                cand
                for cand in candidate_set
                if self.node_info[cand].recompute_ratio == max_recomp_ratio
            )
            # Add to recomputation set
            recomps.append(r_cand)
            # Current candidate
            cand = r_cand
            # Remove from candidate set
            candidate_set.remove(cand)
            # Update recomputation count
            recomp_cnt = self._update_recomps(cand, recomps)
            # Update candidates
            self._update_candidates(cand, recomp_cnt, recomps, candidate_set)
            # Update memory consumption
            mem_consumption -= self.node_info[cand].memory_size
        # Set a flag to indicate that checkpoint selection has been run
        self.is_recomp_selected = True

        return recomps

    def _replace_subsequent_uses_of(self, old_node: fx.Node, new_node: fx.Node) -> None:
        old_node_users = old_node.users
        for node in reversed(self.module.graph.nodes):
            if node == new_node:
                break
            if node in old_node_users:
                node.replace_input_with(old_node, new_node)

    def apply_checkpointing(self, recomps: List[fx.Node]):
        """
        Apply checkpointing to the graph based on the recomputation set.
        """
        # Check that recomputation selection has been run and that all recomputations have a first backward access node
        assert self.is_recomp_selected, (
            "Recomputation selection must be run before applying checkpointing"
        )
        assert all(
            self.node_info[recomp].first_back_access is not None for recomp in recomps
        ), "All recomputations must have a first backward access node"
        recomp_names = [node.name for node in recomps]

        # Print the recomputation set
        print(f"Recomputation set: {recomps}")
        for node in reversed(recomps):
            # Obtain a sub-graph that recomputes the required nodes
            recompute_subgraph = _extract_graph_with_inputs_outputs(
                joint_graph=self.module.graph,
                inputs=self.node_info[node].recomp_srcs,
                outputs=[node],
            )
            print(f"Extracted recomputation sub-graph for {node.name}: ")
            recompute_subgraph.print_tabular()

            # Insert the nodes of the new sub-graph in the old graph before the first
            # backward access of the node to be recomputed.
            node_first_back_access = self.node_info[node].first_back_access
            with self.module.graph.inserting_before(node_first_back_access):
                # Sanity check that the node_first_back_access is after the separator
                assert node_first_back_access is not None, (
                    f"Node {node.name} has no first backward access node"
                )
                assert (
                    self.node_info[node_first_back_access].rank
                    > self.node_info[self.backward_start].rank
                ), f"Node {node.name} is being inserted before the backward pass"

                # initialize name to node mapping
                name_to_node = {node.name: node for node in self.module.graph.nodes}
                name_to_node.pop("output")

                # Create new nodes in the old graph
                for old_node in recompute_subgraph.nodes:
                    if old_node.op == "placeholder" or old_node.op == "output":
                        continue
                    real_old_node = name_to_node[old_node.name]
                    new_node = self.module.graph.node_copy(
                        old_node, arg_transform=lambda arg: name_to_node[arg.name]
                    )
                    print(f"Created new node {new_node.name} for {old_node.name}")

                    # Replace all the uses of the old node with new recomputation node
                    if old_node.name in recomp_names:
                        self._replace_subsequent_uses_of(
                            old_node=real_old_node, new_node=new_node
                        )

                    # Add the new node info
                    n_info = NodeInfo()
                    n_info.rank = self.node_info[node_first_back_access].rank - 0.5
                    n_info.node_type = NodeType.RECOMP
                    n_info.is_recomp = True
                    for input_node in new_node.all_input_nodes:
                        if input_node in self.intermediate_nodes:
                            n_info.first_back_uses.append(input_node)
                    name_to_node[old_node.name] = new_node
                    self.node_info[new_node] = n_info

        # UPDATE ACCESS INFO FOR CORRECT SWAPPING
        for node in recomps:
            # Remove the node from last_forward_uses to prevent it from
            # being swapped out
            last_forward_access = self.node_info[node].last_forward_access
            if last_forward_access is not None:
                self.node_info[last_forward_access].last_forward_uses.remove(node)
                last_forward_access = None

            # Remove the old node from the first_back_access to prevent
            # it from being swapped in when it doesn't exist
            first_back_access = self.node_info[node].first_back_access
            if (
                first_back_access is not None
                and node in self.node_info[first_back_access].first_back_uses
            ):
                self.node_info[first_back_access].first_back_uses.remove(node)
                self.node_info[node].first_back_access = None

        # Lint the graph to ensure it is valid
        self.module.graph.lint()
        # Recompile the module
        self.module.recompile()
        # Update the last uses of the nodes in the interpreter
        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use: dict[fx.Node, fx.Node] = {}
        self.user_to_last_uses: dict[fx.Node, list[fx.Node]] = {}

        def register_last_uses(n: fx.Node, user: fx.Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                self.user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(self.module.graph.nodes):
            for n in node._input_nodes:
                register_last_uses(n, node)

    def get_total_memory_breakdown(self) -> MemStats:
        grad_mem = 0
        act_mem = 0
        other_mem = 0
        param_and_opt_state_mem = self.param_and_opt_state_memory
        for node in self.env.keys():
            if node.op == OP.PLACEHOLDER:
                continue
            node_type = self.node_info[node].node_type
            memory_size = self.node_info[node].memory_size
            if node_type == NodeType.GRAD:
                grad_mem += memory_size
            elif node_type == NodeType.ACT:
                act_mem += memory_size
            else:
                other_mem += memory_size
        mem_stats = MemStats(param_and_opt_state_mem, grad_mem, act_mem, other_mem)
        return mem_stats

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self.param_and_opt_state_memory = torch.cuda.memory_allocated()
        result = super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

        return result

    def run_node(self, node: fx.Node) -> Any:
        if node.op == OP.PLACEHOLDER:
            return super().run_node(node)

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.
        if self.node_info[node].rank > self.node_info[self.backward_start].rank:
            self._swap_in_node(node)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = super().run_node(node)
        end_event.record()
        torch.cuda.synchronize()
        self.env[node] = result
        active_memory = torch.cuda.memory_allocated()

        run_time = start_event.elapsed_time(end_event)
        n_info = self.node_info[node]
        self.node_runtimes.setdefault(node, []).append(run_time)
        n_info.mem_stats = self.get_total_memory_breakdown()
        n_info.peak_total_mem = active_memory + self.swapped_memory

        # Alternate way to measure the memory of the resultant tensor
        if isinstance(result, torch.Tensor):
            size = result.untyped_storage().size()
            element_size = result.untyped_storage().element_size()
            tensor_memory = (
                math.ceil((size * element_size) / _PYTORCH_MIN_ALLOCATE)
                * _PYTORCH_MIN_ALLOCATE
            )
            n_info.memory_size = tensor_memory

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.
        if self.node_info[node].rank < self.node_info[self.forward_end].rank:
            self._swap_out_node(node)

        return result

    def aggregate_stats(self):
        node_start_time = 0.0
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            self.node_info[node].start_time = node_start_time
            self.node_info[node].run_time = mean(self.node_runtimes[node])
            node_start_time += self.node_info[node].run_time

            if node in self.intermediate_nodes:
                self.node_info[node].swap_time = mean(self.node_swap_times[node])

        # Set the inactive time for all nodes
        for node in self.module.graph.nodes:
            last_forward_node = self.node_info[node].last_forward_access
            first_back_node = self.node_info[node].first_back_access
            if last_forward_node is not None and first_back_node is not None:
                self.node_info[node].inactive_time = self.node_info[
                    first_back_node
                ].start_time - (
                    self.node_info[last_forward_node].start_time
                    + self.node_info[last_forward_node].run_time
                )

        # Set a flag to indicate that profiling has been run
        self.is_profiled = True

    def reset_stats(self):
        self.node_runtimes.clear()
        self.node_swap_times.clear()

    def print_stats(self, to_file: str = None):
        headers: List[str] = [
            "Node",
            "Node Type",
            "Target",
            "Size (B)",
            "Avg runtime (ms)",
            "Peak Memory (B)",
            "Swap Time (ms)",
        ]
        node_summaries: List[List[Any]] = []

        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info = self.node_info[node]
            val_list = [
                node.name,
                NodeType(n_info.node_type).name,
                node._pretty_print_target(node.target),
                n_info.memory_size,
                n_info.run_time,
                n_info.peak_total_mem,
            ]
            if node in self.intermediate_nodes:
                val_list.append(n_info.swap_time)
            else:
                val_list.append("")
            node_summaries.append(val_list)
        # print(tabulate.tabulate(node_summaries, headers=headers))
        if to_file:
            with open(to_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(node_summaries)


if __name__ == "__main__":
    print("Executing this file")
