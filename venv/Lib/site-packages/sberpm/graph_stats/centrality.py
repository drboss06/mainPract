from typing import Dict, Any

import numpy as np
import scipy
from scipy.sparse.linalg import eigs

from ..visual._graph import Graph
from ..visual._types import NodeType


def is_graph(graph: Any) -> bool:
    """
    Check if object is graph.
    Parameters
    ----------
    graph: Any
        Arbitrary object
    Returns
    -------
        bool
    """
    return isinstance(graph, Graph)


def degree_centrality(
        G: Graph,
        degree_type: str = "indirect") \
        -> Dict[str, float]:
    """
    Return degree centrality for graph. One can specify type of centrality:
        'in' -> in_degree;
        'out' -> out_degree;
        'indirect' -> ignore direction.

    Parameters
    ----------
    G: Graph
    degree_type: {"in", "out", "indirect"}, default='indirect'

    Returns
    -------
        Dict[str, float] - node_id: node_centrality
    """
    if not is_graph(G):
        raise TypeError(f"Expected Graph, but got {type(G)}")

    nodes = [idx for idx, node in G.nodes.items()
             if node.type not in [NodeType.START_EVENT, NodeType.END_EVENT]]
    if len(nodes) <= 1:
        return {n: 1 for n in nodes}

    if degree_type == "in":
        adj_matrix, compliance = G.get_adjacency_matrix(matrix_type="direct")
        degrees = adj_matrix.sum(axis=0)
    elif degree_type == "out":
        adj_matrix, compliance = G.get_adjacency_matrix(matrix_type="direct")
        degrees = adj_matrix.sum(axis=1)
    elif degree_type == "indirect":
        adj_matrix, compliance = G.get_adjacency_matrix(matrix_type="indirect")
        degrees = adj_matrix.sum(axis=1)
    else:
        raise ValueError('Expected degree_type "in", "out", '
                         f'or "indirect", but got "{degree_type}".')
    degrees = [degrees[compliance[node]] for node in nodes]

    s = 1.0 / (len(nodes) - 1.0)
    centrality = {n: d * s for n, d in zip(nodes, degrees)}
    return centrality


def eigenvector_centrality(
        G: Graph,
        degree_type: str = "indirect"
) -> Dict[str, float]:
    """
    Return eigenvector centrality for graph.
    ith value in eigenvector x (Ax = \lambda x)
        corresponds to eigenvector centrality
    One can specify type of centrality:
        'in' -> in_degree;
        'out' -> out_degree;
        'indirect' -> ignore direction.

    Parameters
    ----------
    G: Graph
    degree_type: {"in", "out", "indirect"}, default='indirect'

    Returns
    -------
        Dict[str, float] - node_id: node_centrality
    """
    if not is_graph(G):
        raise TypeError(f"Expected Graph, but got {type(G)}")
    nodes = [idx for idx, node in G.nodes.items()
             if node.type not in [NodeType.START_EVENT, NodeType.END_EVENT]]
    if degree_type == "in":
        adj_matrix, _ = G.get_adjacency_matrix(matrix_type="direct")
    elif degree_type == "out":
        adj_matrix, _ = G.get_adjacency_matrix(matrix_type="direct")
    elif degree_type == "indirect":
        adj_matrix, _ = G.get_adjacency_matrix(matrix_type="indirect")
    else:
        raise ValueError('Expected degree_type "in", "out", '
                         f'or "indirect", but got "{degree_type}".')
    eigenvalue, eigenvector = eigs(
        adj_matrix, k=1, which="LR", maxiter=100, tol=1e-6,
    )
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * scipy.linalg.norm(largest)
    return dict(zip(nodes, largest / norm))
