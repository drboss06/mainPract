import pickle
from typing import Dict, Tuple

import numpy as np

from ._types import NodeType, GraphType


class Node:
    """
    Represents a node of a graph.

    Parameters
    ----------
    node_id : str
        Unique identification of a node.

    label : str
        Name of a node.

    node_type : str, default=NodeType.task
        Type of a node. All possible node types are specified in NodeType class
    """

    def __init__(self, node_id, label, node_type=NodeType.TASK):
        self.id = node_id
        self.type = node_type
        self.label = label
        self.input_edges = []
        self.output_edges = []
        self.metrics = {}

    def __repr__(self):
        return "Node: " + self.id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def add_input_edge(self, edge):
        """
        Adds an input edge

        Parameters
        ----------
        edge : Edge
            Edge object.
        """
        self.input_edges.append(edge)

    def add_output_edge(self, edge):
        """
        Adds an output edge

        Parameters
        ----------
        edge : Edge
            Edge object.
        """
        self.output_edges.append(edge)

    def add_metric(self, metric_name, metric_value):
        """
        Adds a metric

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_value : float or int
            Value of the metric.
        """
        self.metrics[metric_name] = metric_value


class Edge:
    """
    Represents a directed edge of a graph.

    Parameters
    ----------
    source_node : Node
        Source node object.

    target_node : Node
        Target node object.
    """

    def __init__(self, source_node: Node, target_node: Node):
        self.id = source_node.id + '_' + target_node.id
        self.source_node = source_node
        self.target_node = target_node
        self.label = None
        self.color = None

        self.source_node.add_output_edge(self)
        self.target_node.add_input_edge(self)

        self.metrics = {}

    def __repr__(self):
        return "Edge: " + self.id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Edge) and self.id == other.id

    def add_metric(self, metric_name, metric_value):
        """
        Adds a metric

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_value : float or int
            Value of the metric.
        """
        self.metrics[metric_name] = metric_value


class Graph:
    """
    Represents a directed graph.

    Parameters
    ----------
    type: GraphType
        Type of the graph.
    """

    def __init__(self, type):
        self.type = type
        self.nodes = {}  # dict of {node_id: Node()}
        self.edges = {}  # dict of {(source_node_id, target_node_id): Edge()}
        self.node_metrics_names = set()  # names of existent node metrics
        self.edge_metrics_names = set()  # names of existent edge metrics

    def __repr__(self):
        s = f'Nodes: {[n for n in self.nodes.keys()]}\n'
        s += f'Edges: {[e for e in self.edges.keys()]}\n'
        return s

    def get_adjacency_matrix(
            self,
            matrix_type: str = "direct") \
            -> Tuple[np.array, Dict[str, int]]:
        """
        Get adjacency matrix and compliance {idx_in_adj_matrix: node_id}
            from Graph.

        Parameters
        ----------
        matrix_type: {'direct', 'indirect'}, default='direct'
            If 'direct', take information about edges' directions
            into account. Ignore directions otherwise.

        Returns
        -------
        adj_matrix: np.array
        compliance: Dict[int, int]
        """
        if self.type == GraphType.BPMN or self.type == GraphType.PETRI_NET:
            raise TypeError("Cannot create adjacency matrix for this "
                            "type of graph")
        if matrix_type not in ['direct', 'indirect']:
            raise ValueError('Expected matrix_type "direct", or "indirect", '
                             f'but got "{matrix_type}".')

        nodes = [idx for idx, node in self.nodes.items()
                 if node.type not in [NodeType.START_EVENT, NodeType.END_EVENT]]
        compliance = {node_id: idx for idx, node_id in enumerate(nodes)}
        adj_matrix = np.zeros(shape=(len(compliance), len(compliance)))
        for node_id in compliance:
            for in_edge in self.nodes[node_id].input_edges:
                source_node_id = in_edge.source_node.id
                if source_node_id == NodeType.START_EVENT:
                    continue
                adj_matrix[compliance[source_node_id], compliance[node_id]] += 1
                if matrix_type == "indirect":
                    adj_matrix[compliance[node_id],
                               compliance[source_node_id]] += 1

        return adj_matrix, compliance

    def get_laplacian(self,
                      matrix_type: str = "direct") \
            -> Tuple[np.array, Dict[str, int]]:
        """
        Get Laplacian matrix (L) of graph.
        L = D - A, where D is diagonal matrix of degrees, A - adjacency matrix.
        Parameters
        ----------
        matrix_type: {'direct', 'indirect'}, default='direct'
            If 'direct', take information about edges' directions
            into account. Ignore directions otherwise.

        Returns
        -------
        laplacian_matrix: np.array
        compliance: Dict[int, int]
        """
        A, compliance = \
            self.get_adjacency_matrix(matrix_type=matrix_type)
        D = A.sum(axis=0)
        L = D - A
        return L, compliance

    def add_node(self, node_id, label, node_type=NodeType.TASK):
        """
        Creates a node object and adds it to the graph.
        If node with given parameters has already been created, raises an error.

        Parameters
        ----------
        node_id : str
            Unique identification of a node.

        label: str
            Name of a node.

        node_type : str, default=NodeType.task
            Type of a node. All possible node types are specified in NodeType class
        """
        if node_id not in self.nodes:
            node = Node(node_id, label, node_type=node_type)
            self.nodes[node.id] = Node(node_id, label, node_type=node_type)
        else:
            raise ValueError(f'Node {node_id} was already added to the graph.')

    def add_node_object(self, node):
        """
        Adds a created node object to the graph.
        If node with given id has already been created, raises an error.

        Parameters
        ----------
        node : Node
            Object that represents a node.
        """
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        else:
            raise ValueError(f'Node {node.id} was already added to the graph.')

    def add_edge(self, source_node_id, target_node_id):
        """
        Creates an edge object and adds it to the graph.
        If edge with given parameters has already been created, raises an error.

        Parameters
        ----------
        source_node_id : str
            Id of a source node. Source node object must already be added to the graph.

        target_node_id: str
            Id of a target node. Target node object must already be added to the graph.
        """
        pair = (source_node_id, target_node_id)
        if pair not in self.edges:
            edge = Edge(self.nodes[source_node_id], self.nodes[target_node_id])
            self.edges[pair] = edge
        else:
            raise ValueError(f'Edge {pair} was already added to the graph')

    def get_nodes(self):
        """
        Returns nodes of the graph

        Returns
        ----------
        nodes : list of Node
            Key: node id, value: node object.
        """
        return list(self.nodes.values())

    def get_edges(self):
        """
        Returns edges of the graph

        Returns
        ----------
        nodes : list of Edge
            Key: tuple of source node id and target node id, value: edge object.
        """
        return list(self.edges.values())

    def remove_node_by_id(self, node_id):
        """
        Removes node with given id from the graph if it exists
        (nothing happens if the node does not exist).

        Parameters
        ----------
        node_id: str
            Id of the node that must be removed.
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]

            input_edges = node.input_edges
            output_edges = node.output_edges
            source_nodes = [edge.source_node for edge in input_edges]
            target_nodes = [edge.target_node for edge in output_edges]

            # remove edges (disconnect the other nodes from the one to be removed)
            for edge in input_edges + output_edges:
                self.remove_edge_by_src_trg_id(edge.source_node.id,
                                               edge.target_node.id)
            del self.nodes[node_id]

            # reconnect the other nodes
            for source_node in source_nodes:
                for target_node in target_nodes:
                    self.add_edge(source_node.id, target_node.id)

    def remove_edge_by_src_trg_id(self, source_node_id, target_node_id):
        """
        Removes edge with given ids of  incoming and outgoing nodes
        (nothing happens if the edge does not exist).

        Parameters
        ----------
        source_node_id: str
            Id of the source node. 

        target_node_id
            Id of the target node.
        """
        key = (source_node_id, target_node_id)
        if key in self.edges:
            edge = self.edges[key]
            source_node = edge.source_node
            target_node = edge.target_node
            source_node.output_edges.remove(edge)
            target_node.input_edges.remove(edge)
            del self.edges[key]

    # ---------------------------------------------------------------------------------------------
    # ----------------------------------  Node metrics  -------------------------------------------
    # ---------------------------------------------------------------------------------------------

    def add_node_metric(self, metric_name, metric_data):
        """
        Adds a metric to the 'task' nodes.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_data : dict of {str: number}
            Metric values of 'task' nodes. Key: id of the 'task' node, value: value of the metric.
            If there is at least one node in given metric_data that is not contained in the graph,
            the metric will not be added, an error will be raised.
        """
        self.node_metrics_names.add(metric_name)
        for node_id, metric_value in metric_data.items():
            try:
                self.nodes[node_id].add_metric(metric_name, metric_value)
            except KeyError:
                self.remove_node_metric(metric_name)  # rollback
                raise RuntimeError(
                    f'Failed to add metric "{metric_name}" to the graph: '
                    f'node "{node_id}" does not exist in the graph.')

    def contains_node_metric(self, node_metric_name):
        """
        Checks whether graph contains given node's metric.

        Parameters
        ----------
        node_metric_name : str
            Name of the metric.

        Returns
        ----------
        result : bool
            Returns True if given node's metric's name is preset in graph, False otherwise.
        """
        return node_metric_name in self.node_metrics_names

    def clear_node_metrics(self):
        """
        Remove all the metrics from the nodes in the graph
        """
        self.node_metrics_names.clear()
        for node in self.nodes.values():
            node.metrics = {}

    def remove_node_metric(self, metric_name):
        self.node_metrics_names.remove(metric_name)
        for node in self.nodes.values():
            if metric_name in node.metrics:
                del node.metrics[metric_name]

    # ---------------------------------------------------------------------------------------------
    # ----------------------------------  Edge metrics  -------------------------------------------
    # ---------------------------------------------------------------------------------------------

    def add_edge_metric(self, metric_name, metric_data):
        """
        Adds a metric to the edges.

        Currently works only with graph of type "DFG" only.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_data : dict of {str: number}
            Metric values of edges. Key: tuple(source_node_id, target_node_id), value: value of the metric.
        """
        if self.type == GraphType.DFG:
            num_added_values = self._add_edge_metric_dfg(metric_name,
                                                         metric_data)
        # elif self.type == GraphType.petri_net:
        # num_added_values = self._add_edge_metric_petri(metric_name, metric_data)
        # TODO: problem with existence of several places between same two activities:
        #  act_1 -> p1 -> act2, act_1 -> p2 -> act2. Where must the metric values be put?
        else:
            raise TypeError(
                f'Cannot add edge metrics to graph of type "{self.type}".')

        if num_added_values != 0:
            self.edge_metrics_names.add(metric_name)
        else:
            raise Warning(
                f'Failed to add edge metric "{metric_name}": none of given edges exist in the graph.')

    def contains_edge_metric(self, edge_metric_name):
        """
        Checks whether graph contains given edge's metric.

        Parameters
        ----------
        edge_metric_name : str
            Name of the metric.

        Returns
        ----------
        result : bool
            Returns True if given edge's metric's name is preset in graph, False otherwise.
        """
        return edge_metric_name in self.edge_metrics_names

    def clear_edge_metrics(self):
        """
        Remove all the metrics from the nodes in th e graph
        """
        self.node_metrics_names.clear()
        for node in self.nodes.values():
            node.metrics = {}

    def _add_edge_metric_dfg(self, metric_name, metric_data):
        """
        Adds a metric to the edges in a 'DFG' graph.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_data : dict of {(str, str): number}
            Metric values of edges. Key: id of the 'task' node, value: value of the metric.
        """
        num_metric_values_added = 0
        for edge_tuple, metric_value in metric_data.items():
            if edge_tuple in self.edges:
                self.edges[edge_tuple].add_metric(metric_name, metric_value)
                num_metric_values_added += 1
        return num_metric_values_added

    def _add_edge_metric_petri(self, metric_name, metric_data):
        """
        CURRENTLY NOT USED

        Adds a metric to the edges in a 'Petri-net' graph.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_data : dict of {(str, str): number}
            Metric values of edges. Key: id of the 'task' node, value: value of the metric.
        """
        num_metric_values_added = 0
        for (source_node_id,
             target_node_id), metric_value in metric_data.items():
            edges = self._find_petri_edges(source_node_id, target_node_id)
            if edges is not None:
                for edge in edges:
                    if metric_name not in edge.metrics:
                        edge.add_metric(metric_name, metric_value)
                    else:
                        edge.metrics[metric_name] += metric_value
                num_metric_values_added += 1
        return num_metric_values_added

    def _find_petri_edges(self, source_node_id, target_node_id):
        """
        CURRENTLY NOT USED

        Finds a path between two activities through a place and returns two corresponding edges.

        Path example: source_act -> edge_sp -> place -> edge_pt -> target_act,
        returns: (edge_sp, edge_pt)

        Parameters
        ----------
        source_node_id : str
            Id of source node.
        target_node_id : str
            Id of target node.

        Returns
        ----------
        edges : tuple (Edge, Edge) or None
            Tuple of two edges if the path has been found, none otherwise.
        """
        source_node = self.nodes[source_node_id]
        for edge_source_place in source_node.output_edges:
            place_node = edge_source_place.target_node
            for edge_place_target in place_node.output_edges:
                if edge_place_target.target_node.id == target_node_id:
                    return edge_source_place, edge_place_target
        return None

    def save(self, file_name):
        """
        Save the graph to a file.

        Parameters
        ----------
        file_name: str
            Name of the file.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)


def load_graph(file_name):
    """
    Loads a graph from the file.

    Returns
    -------
    graph: Graph
        Graph.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)

    # ---------------------------------------------------------------------------------------------
    # ----------------------------- 'Create graph' methods  ---------------------------------------
    # ---------------------------------------------------------------------------------------------


def create_petri_net():
    return Graph(GraphType.PETRI_NET)


def create_dfg():
    return Graph(GraphType.DFG)


def create_bpmn():
    return Graph(GraphType.BPMN)
