from enum import Enum
from typing import Dict
import torch
import math
import torch.fx as fx
from typing import Dict, Any, List

# Minimum memory allocated by PyTorch, change according to your device type
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
    STATE = 3
    OTHER = 4 

class NodeInfo:
    rank: int = 0
    node_type: NodeType = None
    run_time: float = 1.0
    peak_mem: int = 0
    active_mem: int  = 0
    memory_size: int = 0
    cpu_ref: torch.Tensor = None
    last_forward_access: fx.Node = None
    first_back_access: fx.Node = None
    last_forward_uses: List[fx.Node] = []
    first_back_uses: List [fx.Node] = []
    #You can add more attributes to this class for applying the recomputation algorithm




# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        self.module = module
        self.node_info: Dict[fx.Node, NodeInfo] = {}
        self.intermediate_nodes:List[fx.Node] = []
        self.node_runtimes:Dict[fx.Node, List[float]] = {}
        self.node_swap_times:Dict[fx.Node, List[float]] = {}
        self.forward_end: fx.Node = None
        self.backward_start: fx.Node = None

        # First we find all the deatch nodes and tag_grad nodes and remove them
        for node in self.module.graph.nodes:
            if node.target == torch.ops.aten.detach.default:
                input_node = node.all_input_nodes[0]
                node.replace_all_uses_with(input_node)
                if len(node.users) == 0:
                    self.module.graph.erase_node(node)
            if (node.target == torch.ops.dummy.tag_grad.default):
                grad_node = node.all_input_nodes[0]
                node.replace_all_uses_with(grad_node)
                if len(node.users) == 0:
                    self.module.graph.erase_node(node)

        rank = 0
        for node in self.module.graph.nodes:
            n_info = NodeInfo()
            n_info.rank = rank
            # Initially set the node types of all nodes to other
            n_info.node_type = NodeType.OTHER
            rank += 1
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

                assert len(param_adam_args) == len(grad_adam_args), "Unequal number of params and gradients"

                for param in param_adam_args:
                    assert isinstance(
                        param, fx.Node
                    ), "Expected param to be an fx.Node instance"
                    assert (
                        param.op == OP.PLACEHOLDER
                    ), "Expected all params nodes to be of type PLACEHOLDER"
                    self.node_info[param].node_type = NodeType.PARAM

                for grad in grad_adam_args:
                    assert isinstance(
                        grad, fx.Node
                    ), "Expected grad to be an fx.Node instance"
                    self.node_info[grad].node_type = NodeType.GRAD    

        for node in self.module.graph.nodes:
            if (
                node.op != OP.PLACEHOLDER
                and self.node_info[node].rank < self.node_info[self.forward_end].rank
            ):
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
                    self.node_info[node].node_type = NodeType.ACT
                    self.node_info[last_forward].last_forward_uses.append(node)
                    self.node_info[first_backward].first_back_uses.append(node)
                    self.node_info[node].first_back_access = first_backward
                    self.node_info[node].last_forward_access = last_forward                   

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
                cpu_ref = torch.zeros(
                    tensor.size(), dtype=tensor.dtype, layout=tensor.layout
                ).pin_memory()
            assert cpu_ref.is_pinned, f"CPU ref is not pinned for {o_node.name}"

            swap_start_event = torch.cuda.Event()
            swap_end_event = torch.cuda.Event()

            swap_start_event.record()
            cpu_ref = cpu_ref.copy_(tensor, False)
            swap_end_event.record()

            torch.cuda.synchronize()
            o_info.cpu_ref = cpu_ref
            self.env[o_node] = cpu_ref
            del tensor
            tensor = None
            cpu_ref = None
            swap_time = swap_start_event.elapsed_time(swap_end_event)
            self.node_swap_times.setdefault(o_node, [])
            self.node_swap_times[o_node].append(swap_time)

    def _swap_in_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be prefetched
        # 2) Retrieve their CPU reference (assert if it resides in pinned
        #    memory)
        # 3) Copy the tensor to GPU memory and add it to the Interpreter
        #    environment
        # 4) Update the state of intermediate tensor in NodeInfo
        nodes_to_fetch = self.node_info[node].first_back_uses
        for p_node in nodes_to_fetch:
            p_info = self.node_info[p_node]
            cpu_ref = p_info.cpu_ref
            assert isinstance(cpu_ref, torch.Tensor), f"CPU ref is not a tensor for {p_node.name}"
            assert cpu_ref.is_pinned, f"CPU ref is not pinned for {p_node.name}"

            swap_start_event = torch.cuda.Event()
            swap_end_event = torch.cuda.Event()

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
            swap_time = swap_start_event.elapsed_time(swap_end_event)
            self.node_swap_times.setdefault(p_node, [])
            self.node_swap_times[p_node].append(swap_time)

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> torch.Any:
        self.param_and_opt_state_memory: int = torch.cuda.memory_allocated()
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, node: fx.Node) -> Any:
        if node.op == OP.PLACEHOLDER:
            return super().run_node(node)
        

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.
        self._swap_in_node(node)

        # you can start measuring the run-time of a node here

        start_event = torch.cuda.Event()
        end_event = torch.cuda.Event()
        start_event.record()
        result = super().run_node(node)
        end_event.record()
        torch.cuda.synchronize()
        run_time = start_event.elapsed_time(end_event)
        
        self.node_info[node].active_mem = torch.cuda.memory_allocated()
        self.node_runtimes.setdefault(node, [])
        self.node_runtimes[node].append(run_time)
        # you can end measuring the run-time of a node here
        # HINT: Use torch.cuda.Events for doing time measurements of operations.

        #Measure the memory of the resultant tensor if the node is an intermediate node
        if node in self.intermediate_nodes:
            int_n_info = self.node_info[node]
            assert isinstance(result, torch.Tensor)
            size = result.untyped_storage().size()
            element_size = result.untyped_storage().element_size()
            tensor_memory = (math.ceil((size * element_size) / _PYTORCH_MIN_ALLOCATE)
            * _PYTORCH_MIN_ALLOCATE)
            int_n_info.memory_size = tensor_memory

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.
        self._swap_out_node(node)

        return result
    
    def summarize(self):
        pass



    

if __name__ == "__main__":
    print("Executing this file")
