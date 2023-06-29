from copy import deepcopy

from ...visual._graph import Node, Graph
from ...visual._types import GraphType, NodeType


def petri_net_to_bpmn(petri_net) -> Graph:
    """

    Converts a given petri net to a bpmn graph.

    Parameters
    ----------
    petri_net: Graph
        Graph of type 'Petri-Net'.

    Returns
    -------
    bpmn_graph: Graph
        Graph of type 'BPMN'.
    """
    graph = deepcopy(petri_net)
    graph.type = GraphType.BPMN

    gateway_node_number = 0
    gateway_nodes = []

    for node in graph.get_nodes():
        source_nodes = [edge.source_node for edge in node.input_edges]  # edges from places and transitions
        target_nodes = [edge.target_node for edge in node.output_edges]  # edges from places and transitions
        in_num = len(source_nodes)
        out_num = len(target_nodes)

        if in_num > 1 >= out_num:
            gateway_node, gateway_node_number = create_gateway_node(node.type, gateway_node_number)
            gateway_nodes.append(gateway_node)
            insert_node(graph, gateway_node, source_nodes, [node])
        elif in_num <= 1 < out_num:
            gateway_node, gateway_node_number = create_gateway_node(node.type, gateway_node_number)
            insert_node(graph, gateway_node, [node], target_nodes)
            gateway_nodes.append(gateway_node)
        elif in_num > 1 and out_num > 1:
            if node.type == NodeType.TASK:
                # Add gateways before and after the transition
                gateway_node_1, gateway_node_number = create_gateway_node(node.type, gateway_node_number)
                gateway_node_2, gateway_node_number = create_gateway_node(node.type, gateway_node_number)
                gateway_nodes.append(gateway_node_1)
                gateway_nodes.append(gateway_node_2)
                insert_node(graph, gateway_node_1, source_nodes, [node])
                insert_node(graph, gateway_node_2, [node], target_nodes)
            else:  # place
                # Replace the place with the gateway
                gateway_node, gateway_node_number = create_gateway_node(node.type, gateway_node_number)
                gateway_nodes.append(gateway_node)
                graph.remove_node_by_id(node.id)
                insert_node(graph, gateway_node, source_nodes, target_nodes)

    # Remove places
    for node in graph.get_nodes():
        if node.type == NodeType.PLACE:
            graph.remove_node_by_id(node.id)

    return graph


def insert_node(graph, node_to_insert, source_nodes, target_nodes):
    """
    Adds a new node to the graph and makes two groups of connected to each other nodes
    connected through the given node.

    Example: transforms "(Node_1, Node_2) -> (Node_3, Node_4)" into
    "(Node_1, Node_2) -> Given_Node -> (Node_3, Node_4)".

    Parameters
    ----------
    graph: Graph
        Graph.

    node_to_insert: Node
        Node that will be added to the graph.

    source_nodes: list of Node
        Nodes that will be connected with the node_to_insert: source_node -> node_to_insert.

    target_nodes: list of Node
        Nodes that will be connected with the node_to_insert: node_to_insert -> target_node.
    """
    # Remove edges (disconnect nodes in the two groups from each other)
    for source_node in source_nodes:
        for target_node in target_nodes:
            graph.remove_edge_by_src_trg_id(source_node.id, target_node.id)
    # Add given node to the graph
    graph.add_node_object(node_to_insert)
    # Connect the node to the other ones.
    for source_node in source_nodes:
        graph.add_edge(source_node.id, node_to_insert.id)

    for target_node in target_nodes:
        graph.add_edge(node_to_insert.id, target_node.id)


def create_gateway_node(current_node_type, gateway_node_number):
    """
    Creates the gateway node.

    Parameters
    ----------
    current_node_type: {NodeType.task, NodeType.place, NodeType.startevent, NodeType.endevent}
        If the given node represents a task, a parallel gateway will be created, otherwise an exclusive gateway.

    gateway_node_number: int
        Unique number of the new gateway node.

    Returns
    -------
    gateway_node: Node
        Gateway node.

    gateway_node_number:
        Unique number of the next new gateway node if it will be created.
    """
    gateway_type = NodeType.PARALLEL_GATEWAY if current_node_type == NodeType.TASK else NodeType.EXCLUSIVE_GATEWAY
    gateway_node = Node(f'gateway_{gateway_node_number}', label='', node_type=gateway_type)
    gateway_node_number += 1
    return gateway_node, gateway_node_number
