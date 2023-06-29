from typing import Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd

from ._graph import Graph


def calc_nodes_colors_by_metric(graph, node_style_metric,
                                darkest_color=100,
                                lightest_color=250
                                ) -> Dict[str, str]:
    """
    Calculate nodes' colours according to given metric.

    Parameters
    -------------
    graph: Graph
        Graph object.

    node_style_metric
        Name of the metric.

    darkest_color: int, default=100
        The number between 0 and 255. 0 is the darkest color possible.

    lightest_color: int, default=250
        The number between 0 and 255. 255 is the lightest color possible.

    Returns
    -------------
    node_color_dict: dict of (str: str)
        Key: node id, value: its colour according to the metric.
        If something goes wrong, an empty dict is returned.
    """
    if node_style_metric is None:
        return {}
    if not graph.contains_node_metric(node_style_metric):
        print(f'WARNING: graph does not contain node metric "{node_style_metric}". '
              f'Nodes will have the same colour.')
        return {}
    nodes = graph.get_nodes()
    metric_values = [node.metrics[node_style_metric] for node in nodes if node_style_metric in node.metrics]
    if any(np.isnan(metric_values)):
        print(f"WARNING: metric \"{node_style_metric}\" contains None values, "
              f"impossible to use it for changing the nodes' style")
        return {}

    node_color_dict = {}
    min_value = min(metric_values)
    max_value = max(metric_values)
    if min_value == max_value:
        return {}

    for node in nodes:
        if node_style_metric in node.metrics:
            node_metric_value = node.metrics[node_style_metric]
            node_color_int = int(
                lightest_color - (lightest_color - darkest_color) * (node_metric_value - min_value) / (
                        max_value - min_value))
            node_color_dict[node.id] = _get_hex_color(node_color_int)
        else:
            node_color_dict[node.id] = None
    return node_color_dict


def calc_edges_widths_by_metric(graph, edge_style_metric,
                                min_width=0.1,
                                max_width=5) -> Dict[Tuple[str, str], float]:
    """
    Calculate edges' width according to given metric.

    Parameters
    -------------
    graph: Graph
        Graph object.
    edge_style_metric
        Name of the metric.

    min_width: float, default=0.1
        Minimum edge width.

    max_width: float, default=0.1
        Maximum edge width.

    Returns
    -------------
    node_color_dict: dict of ((str, str): float)
        Key: edge name, value: its width according to the metric.
        If something goes wrong, an empty dict is returned.
    """
    if edge_style_metric is None:
        return {}
    if not graph.contains_edge_metric(edge_style_metric):
        print(f'WARNING: graph does not contain edge metric "{edge_style_metric}". '
              f'Edges will have the same width.')
        return {}
    metric_values = [edge.metrics[edge_style_metric] for edge in graph.get_edges() if
                     edge_style_metric in edge.metrics]
    if any(np.isnan(metric_values)):
        print(f"WARNING: metric \"{edge_style_metric}\" contains None values, "
              f"impossible to use it for changing the edges' style")
        return {}

    edge_width_dict = {}
    min_value = min(metric_values)
    max_value = max(metric_values)
    if min_value == max_value:
        return {}

    for n1n2, edge in graph.edges.items():
        if edge_style_metric in edge.metrics:
            edge_metric_value = edge.metrics[edge_style_metric]
            score = (edge_metric_value - min_value) / (max_value - min_value)
            edge_width_dict[n1n2] = min_width + (max_width - min_width) * score
    return edge_width_dict


def _get_hex_color(int_color):
    """
    Transform color to hexadecimal representation.

    Parameters
    -------------
    int_color: int
        Number from 0 to 255.

    Returns
    -------------
    hex_string: str
        Hexadecimal color. Example: "#AA5500".
    """
    left_r = _get_hex_string(int(int_color) // 16)
    right_r = _get_hex_string(int(int_color) % 16)

    left_g = _get_hex_string(int(int_color) // 16)
    right_g = _get_hex_string(int(int_color) % 16)

    left_b = _get_hex_string(int(int_color) // 16)
    right_b = _get_hex_string(int(int_color) % 16)

    res = "#" + left_r + right_r + left_g + right_g + left_b + right_b
    return res


def _get_hex_string(num):
    """
    Returns an hexadecimal string corresponding to a given number.

    Parameters
    -------------
    num: int
        Number, must be one of {0, 1, 2,..., 13, 14, 15}.

    Returns
    -------------
    hex_string: str
        Hexadecimal string.
    """
    if num < 10:
        return str(num)
    else:
        return ['A', 'B', 'C', 'D', 'E', 'F'][num - 10]


def add_metrics_to_node_label(label: str,
                              metrics: Dict[str, float],
                              newline: Optional[str] = '\\n') -> str:
    """
    Adds information about node's metrics to its label.

    Parameters
    -------------
    label: str
        Label of the node.

    metrics: dict of {str: number}
        Metrics of this node.

    newline: str, default='\\n'
        New line symbol.

    Returns
    -------------
    label: str
        Modified node's label.
    """
    for metric_name, metric_value in metrics.items():
        label += newline + metric_name + ': ' + str(round(metric_value, 3))
    return label


def add_metric_to_node_label(label: str,
                             metric_name: str,
                             metric_value: float,
                             newline: Optional[str] = '\\n') -> str:
    """
    Adds information about node's metrics to its label.

    Parameters
    -------------
    label: str
        Label of the node.

    metric_name: str

    metric_value: float

    newline: str, default='\\n'
        New line symbol.

    Returns
    -------------
    label: str
        Modified node's label.
    """
    label += newline + metric_name + ': ' + str(round(metric_value, 3))
    return label

def _get_hex_red_green_color(red, green) -> str:
    """
    Transform color to hexadecimal representation.

    Parameters
    -------------
    red: int
        Number from 0 to 255.

    green: int
        Number from 0 to 255.

    Returns
    -------------
    hex_string: str
        Hexadecimal color. Example: "#AA5500".
    """
    left_r = _get_hex_string(int(red) // 16)
    right_r = _get_hex_string(int(red) % 16)

    left_g = _get_hex_string(int(green) // 16)
    right_g = _get_hex_string(int(green) % 16)

    left_b = _get_hex_string(0)
    right_b = _get_hex_string(0)

    res = "#" + left_r + right_r + left_g + right_g + left_b + right_b
    return res
