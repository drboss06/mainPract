import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..visual._graph import create_dfg


# ------------------------------------------------------------------------------
# ----------  Connected components and strongly connected components  ----------
# ------------------------------------------------------------------------------

def get_strongly_connected_components(graph):
    """
    Finds strongly connected components in a given graph.

    Parameters
    ----------
    graph: Graph
        Graph.

    Returns
    -------
    components: list of set of str
        List of components.
    """
    return _get_connected_components(graph, 'strong')


def get_weakly_connected_components(graph):
    """
    Finds connected components in a given graph.

    Parameters
    ----------
    graph: Graph
        Graph.

    Returns
    -------
    components: list of set of str
        List of components.
    """
    return _get_connected_components(graph, 'weak')


def _get_connected_components(graph, connection):
    """
    Finds connected components/strongly connected components in a given graph.

    Parameters
    ----------
    graph: Graph
        Graph.

    connection: {'weak', 'strong'}
        Whether to find connected components or strongly connected components.

    Returns
    -------
    components: list of set of str
        List of components. A component is a set of node names.
    """
    node2ind = {node: num for num, node in enumerate(graph.nodes.keys())}
    ind2node = {num: node for node, num in node2ind.items()}
    matrix = np.zeros((len(node2ind), len(node2ind)))

    for n1, n2 in graph.edges.keys():
        matrix[node2ind[n1], node2ind[n2]] = 1
    matrix = csr_matrix(matrix)

    comp_number, comps = connected_components(csgraph=matrix, directed=True, connection=connection,
                                              return_labels=True)
    components = [set() for _ in range(comp_number)]
    for node_ind, node_comp in enumerate(comps):
        components[node_comp].add(ind2node[node_ind])

    return components


# ------------------------------------------------------------------------------
# -----------------------  Operations with the graph  --------------------------
# ------------------------------------------------------------------------------

def cut_graph(graph, start_nodes, end_nodes, node_groups):
    """
    Cuts the graph into several subgraphs.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    node_groups: list of set of str
        Each group of nodes will be a new subgraph.

    Returns
    -------
    se_subgraphs: list of [graph, start_nodes, end_nodes], where

        graph: Graph
            Graph.

        start_nodes: set of str
            Start nodes of the graph.

        end_nodes: set of str
            Start nodes of the graph.

    """
    node_groups = list(node_groups)
    subgraphs = [create_dfg() for _ in range(len(node_groups))]
    se_subgraphs = []
    for i, node_group in enumerate(node_groups):
        snodes = set([node for node in start_nodes if node in node_group])
        enodes = set([node for node in end_nodes if node in node_group])
        subgraph = subgraphs[i]
        for node in graph.nodes.keys():
            if node in node_group:
                subgraph.add_node(node, node)

        for n1, n2 in graph.edges.keys():
            if n1 in node_group and n2 in node_group:
                subgraph.add_edge(n1, n2)

        # Set start and end nodes
        if len(node_group) == 1 and len(subgraph.edges) != 0:  # self loop
            snodes.add(list(node_group)[0])
            enodes.add(list(node_group)[0])
        else:
            for node in node_group:
                for edge in graph.nodes[node].input_edges:
                    if edge.source_node.id not in node_group:
                        snodes.add(node)
                        break
                for edge in graph.nodes[node].output_edges:
                    if edge.target_node.id not in node_group:
                        enodes.add(node)
                        break

        se_subgraphs.append([subgraph, snodes, enodes])

    return se_subgraphs


def create_graph_without_nodes(graph, nodes_to_exclude):
    """
    Creates a copy of the given graph excluding the given nodes.

    Parameters
    ----------
    graph: Graph
        Graph.

    nodes_to_exclude: Iterable of str
        Nodes that must not be present in the new graph.

    Returns
    -------
    new_graph: Graph
        Graph.
    """
    new_graph = create_dfg()
    for node in graph.nodes.keys():
        if node not in nodes_to_exclude:
            new_graph.add_node(node, node)
    for n1, n2 in graph.edges.keys():
        if n1 not in nodes_to_exclude and n2 not in nodes_to_exclude:
            new_graph.add_edge(n1, n2)
    return new_graph


def create_inverted_graph(graph):
    """
    Creates the inverted graph to the given one.

    The inverted graph:
    - has the same nodes as the given graph
    - has either the two edges between nodes A anb B (A->B and B->A) or none (A-/>B and B-/>A)
    - has no edges between nodes A anb B only if the given graph has two edges between them (A->B and B->A),
      if the given graph has no edges or only one edge, the inverted graph has two edges between them.

    Parameters
    ----------
    graph: Graph
        Graph

    Returns
    -------
    inv_graph: Graph
        Inverted graph.
    """
    inv_graph = create_dfg()
    nodes = list(graph.nodes.keys())
    for node in nodes:
        inv_graph.add_node(node, node)

    for i in range(len(nodes)):
        n1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            n2 = nodes[j]
            if not ((n1, n2) in graph.edges and (n2, n1) in graph.edges):
                inv_graph.add_edge(n1, n2)
                inv_graph.add_edge(n2, n1)

    return inv_graph


def check_each_node_group_has_start_end_nodes(node_groups, start_nodes, end_nodes):
    """
    Checks whether each group of nodes has at least one start and at least one end nodes.

    Parameters
    ----------
    node_groups: list of set of str
        List of node groups.

    start_nodes: set of str
        Start nodes of a graph.

    end_nodes: set of str
        End nodes of a graph.

    Returns
    -------
    result: bool
        True if each group of nodes has at least one start and at least one end nodes, False otherwise.
    """
    for group in node_groups:
        if len(start_nodes.intersection(group)) == 0 or (end_nodes.intersection(group)) == 0:
            return False
    return True


def get_graph_with_grouped_nodes(graph, start_nodes, end_nodes, node_groups):
    """
    Creates a graph that is a copy of the given one with its nodes
    grouped accordingly to the node_groups list.

    Each new node (group of real nodes) of the new graph will have a name
    that corresponds to the index of the group in node_groups list.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        End nodes of the graph.

    node_groups: list of set of str
        List of node groups. All the nodes in each group will be united
        and will form a new separate node.
    """
    node2group = {node: str(group) for group, nodes in enumerate(node_groups) for node in nodes}

    new_graph = create_dfg()
    for group in range(len(node_groups)):
        new_graph.add_node(str(group), str(group))
    for n1, n2 in graph.edges.keys():
        comp1 = node2group[n1]
        comp2 = node2group[n2]
        if comp1 != comp2 and (comp1, comp2) not in new_graph.edges:
            new_graph.add_edge(comp1, comp2)

    new_start_nodes = set([node2group[node] for node in start_nodes])
    new_end_nodes = set([node2group[node] for node in end_nodes])

    return new_graph, new_start_nodes, new_end_nodes
