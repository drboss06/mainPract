import warnings
from typing import Tuple, Dict, List

import pydotplus
from pydotplus import InvocationException

from ...visual._graph import Graph
from ...visual._types import NodeType


def calc_coords_graphviz(bpmn_graph: Graph) -> Tuple[Dict[str, Dict],
                                                     Dict[Tuple[str, str], List]]:
    """
    Calculates coordinates of the graph.

    Parameters
    ----------
    bpmn_graph: Graph

    Returns
    -------
    node_params_dict: dict
        Parameters of nodes. Keys are the same as bpmn_graph.nodes.keys(),
        values include: 'label', 'height', 'width', 'x', 'y'.

    edge_pos_dict: dict
        Waypoints (coordinates) of edges. Keys are the same as bpmn_graph.edges.keys(),
        value is a list of tuples (x, y).
    """
    dot, gvname_to_id = bpmn_to_dot(bpmn_graph)

    # Create pydotplus graph with coordinates
    dot.set('rankdir', 'LR')
    dot.set('splines', 'spline')
    try:
        dot_data = dot.create(prog='dot', format='dot')
    except InvocationException:
        warnings.simplefilter('always', RuntimeWarning)
        warnings.warn("Impossible to create orthogonal edges, splines will be created instead.", RuntimeWarning)
        dot.set('splines', 'spline')
        dot_data = dot.create(prog='dot', format='dot', )
    dot_coords = pydotplus.graph_from_dot_data(dot_data)

    # Get coordinates and other params
    # Node params
    node_params_dict = {}
    for graph_node in dot_coords.get_node_list():
        gvname = graph_node.get_name()

        label = graph_node.__get_attribute__('label')
        height = graph_node.__get_attribute__('height')
        width = graph_node.__get_attribute__('width')
        pos = graph_node.__get_attribute__('pos')

        inc_size_param = 72.72
        if pos is not None:  # not to include 'system' nodes
            height, width = float(height), float(width)
            height, width = height * inc_size_param, width * inc_size_param
            pos = tuple(float(el) for el in pos[1:-1].split(','))  # two float numbers
            x, y = pos[0] - width / 2, pos[1] - height / 2  # change coords to node's center
            gvname = remove_quotes(gvname)
            label = remove_quotes(label)[:-1]  # and remove last space
            real_node_id = gvname_to_id[gvname]
            node_params_dict[real_node_id] = dict(label=label, height=height, width=width, x=x, y=y)

    # Edge params
    edge_pos_dict = {}
    for graph_edge in dot_coords.get_edge_list():
        source_gvname = remove_quotes(graph_edge.get_source())
        dest_gvname = remove_quotes(graph_edge.get_destination())

        pos = graph_edge.__get_attribute__('pos')
        pos = parse_pos(pos)
        real_s_id = gvname_to_id[source_gvname]
        real_d_id = gvname_to_id[dest_gvname]
        edge_pos_dict[(real_s_id, real_d_id)] = pos

    return node_params_dict, edge_pos_dict


def bpmn_to_dot(bpmn_graph: Graph) -> Tuple[pydotplus.Dot,
                                     Dict[str, str]]:
    """
    Transform given bpmn graph to a graph object that can be visualized.

    Parameters
    ----------
    bpmn_graph: Graph
        BPMN graph.

    Returns
    ----------
    dot: pydotplus.Dot
        Representation of the graph that can be visualized.

    gvname_to_id: dict
    """
    dot = pydotplus.Dot()
    pydot_plus_node_dict = {}
    gvname_to_id = {}
    for i, node in enumerate(bpmn_graph.nodes.values()):
        # * node names have a space at the end to avoid graphviz errors
        gvname = str(i) + ' '

        if node.type == NodeType.START_EVENT:
            n = pydotplus.Node(name=gvname, label='', shape='circle', fillcolor='green')
        elif node.type == NodeType.END_EVENT:
            n = pydotplus.Node(name=gvname, label='', shape='circle', fillcolor='red')
        elif node.type == NodeType.TASK:
            n = pydotplus.Node(name=gvname, label=node.label + ' ', shape='box')
        elif node.type == NodeType.PARALLEL_GATEWAY:
            n = pydotplus.Node(name=gvname, label='+', shape='diamond')
        elif node.type == NodeType.EXCLUSIVE_GATEWAY:
            n = pydotplus.Node(name=gvname, label='x', shape='diamond')
        else:
            raise TypeError(f'Node of type "{node.type}" is not expected to be in a BPMN graph.')
        pydot_plus_node_dict[node.id] = n
        gvname_to_id[gvname] = node.id
        dot.add_node(n)

    for edge in bpmn_graph.get_edges():
        e = pydotplus.Edge(src=pydot_plus_node_dict[edge.source_node.id],
                           dst=pydot_plus_node_dict[edge.target_node.id])
        dot.add_edge(e)

    return dot, gvname_to_id


def remove_quotes(s: str) -> str:
    """
    Removes quotes from a string.
    Transform str: "some text" -> some text.
    """
    return s[1:-1]


def parse_pos(pos: str) -> List[Tuple[float, float]]:
    pos = pos[1:-1]  # remove '"'
    pos = pos.replace('e,', '')
    pairs = [pair for pair in pos.split(' ')]
    # the first pair (with 'e') is actually the destination point
    pos_list = []
    for pair in pairs[1:]:
        x_y = pair.split(',')
        # if there is a new line, '\' appears
        pos_list.append((float(x_y[0].replace("\\", '')), float(x_y[1].replace("\\", ''))))
    x_y = pairs[0].split(',')
    pos_list.append((float(x_y[0]), float(x_y[1])))
    return pos_list
