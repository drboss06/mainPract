from typing import Tuple, Dict, List

from ...visual import MlPainter
from ...visual._graph import Graph


def calc_coords_grandalf(bpmn_graph: Graph) -> Tuple[Dict[str, Dict],
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
    painter = MlPainter()
    painter.apply(bpmn_graph,
                  vertical=False)
    inc_size_param = 5
    node_params_dict = {}
    for name, node in bpmn_graph.nodes.items():
        label = painter.nodes[name]['label']
        pmin, pmax = painter.nodes[name]['bbox_coords']
        height = (pmax.y - pmin.y) * inc_size_param
        width = (pmax.x - pmin.x) * inc_size_param
        pos = [el * inc_size_param for el in painter.nodes[name]['coords'].xy()]
        x, y = pos[0] - width / 2, pos[1] - height / 2
        node_params_dict[name] = dict(label=label, height=height, width=width, x=x, y=y)

    edge_pos_dict = {}
    for n1n2, edge in bpmn_graph.edges.items():
        edge_pos_dict[n1n2] = [[el * inc_size_param for el in p.xy()] for p in painter.edges[n1n2]['coords']]
    return node_params_dict, edge_pos_dict
