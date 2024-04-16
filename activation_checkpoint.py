import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs


# We define a custom function that takes in two weight matrices that require
# gradients to be computed and an input data matrix. The function returns the
# gradients of the weight matrices with respect to the loss (sum in our
# example). NOTE: The custom function mimics a simple two layer liner neural
# network with relu activation functions and a sum loss function.
def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.mm(w1, x)
    z = nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = nn.functional.relu(z)
    z = z.sum()
    z.backward()
    return w1.grad, w2.grad


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[fx.Node, str]:
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def activation_checkpointing(gm: fx.GraphModule) -> fx.GraphModule:
    # NOTE: You need to create the function for your project and call it inside
    # the graph_transformation function after performing graph profiling.

    # In this example we are going to recompute one of the relu activations for the
    # backward pass instead of saving it. We know from our custom function
    # that we have 4 intermeidate nodes: ['mm', 'relu', 'mm_1', 'relu_1']

    # So the intermediate node to recompute is: ['relu'] and
    # intermediate nodes to checkpoint (retain) are: ['mm', 'mm_1', 'relu_1']

    # Nodes required to recompute 'relu' are ['mm']
    # First back use is at node 't'

    # NOTE: For your project, you will use GraphProfiler to identify the
    # intermediate nodes, their first back access, last forward access and
    # then MuTWO's algorithm to select the intermediate 'nodes_to_recompute' and
    # checkpoint (retain). The 'nodes_required_to_recompute' any of the
    # intermediate nodes MUST be a combination of the placeholder nodes and the
    # intermediate nodes that are checkpointed.

    name_to_node = get_name_to_node_map(gm)
    first_back_access = name_to_node["t"]
    node_to_recompute = [name_to_node["relu"]]
    nodes_required_to_recompute = [name_to_node["mm"]]

    # Obtain a sub-graph that recomputes the required nodes
    recompute_subgraph = _extract_graph_with_inputs_outputs(
        joint_graph=gm.graph,
        inputs=nodes_required_to_recompute,
        outputs=node_to_recompute,
    )
    print("Extracted recomputation sub-graph: ")
    recompute_subgraph.print_tabular()

    # Insert the nodes of the new sub-graph in the old graph before the first
    # backward access of the node to be recomputed.
    with gm.graph.inserting_before(first_back_access):
        for n in recompute_subgraph.nodes:
            if n.op == "placeholder" or n.op == "output":
                continue
            # Copy the nodes of the new sub-graph to old graph and transform its
            # inputs to match the old-graph
            new_node = gm.graph.node_copy(
                n, arg_transform=lambda x: name_to_node[x.name]
            )
            old_node = node_to_recompute[0]
            # Replace all the uses of the old node with new recomputation node
            replace_subsequent_uses_of(gm.graph, old_node=old_node, new_node=new_node)
            # Add the recomputed node to our mapping
            name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()
    return gm


if __name__ == "__main__":
    # Create two weight matrices that require gradients and one input data matrix
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    # Create a graph module by tracing the the custom function with the given inputs
    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the traced function
    # NOTE: We have already captured the backward operations during tracing
    # hence we are executing in no grad mode
    with torch.no_grad():
        old_grads = graph_module(w1, w2, x)

    # Apply the activation checkpointing algorithm (check new node 'relu_2')
    new_graph_module = activation_checkpointing(graph_module)
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the activation
    # checkpointed function to recalculate them
    with torch.no_grad():
        new_grads = new_graph_module(w1, w2, x)

    # Verify that gradients produced with activation checkpointing equal the
    # ones obtained earlier with no optimization.
    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))
