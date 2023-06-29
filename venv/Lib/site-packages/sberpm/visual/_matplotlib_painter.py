import copy
from typing import List, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
from grandalf import graphs as grand_graph
from grandalf.layouts import SugiyamaLayout, VertexViewer
from grandalf.routing import EdgeViewer, _round_corners, ROUND_AT_DISTANCE

from . import utils
from ._graph import Graph, Node, Edge
from ._types import NodeType, GraphType
from ..miners._inductive_miner import ProcessTreeNode, ProcessTreeNodeType


def route_with_rounded_corners(e: grand_graph.Edge) -> None:
    """
    Algorithm for making rounded corners
    copied from grandalf library.

    Parameters
    ----------
    e: Edge
        Edge object in grandalf library.

    Returns
    -------
    None
    """
    pts = e.view._pts
    new_pts = _round_corners(pts, round_at_distance=ROUND_AT_DISTANCE)
    pts[:] = new_pts[:]


class Point:
    """
    Point class.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point({self.x}, {self.y})'

    def upd(self, x: float = None, y: float = None) -> 'Point':
        """
        Updates point coordinates.

        Parameters
        ----------
        x: float, default=None
        y: float, default=None

        Returns
        -------
        self
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        return self

    def add(self, p: 'Point') -> 'Point':
        self.x += p.x
        self.y += p.y
        return self

    def neg(self) -> 'Point':
        return Point(-self.x, -self.y)

    def updx(self, x: float) -> 'Point':
        """
        Updates point x-coordinate.

        Parameters
        ----------
        x: float

        Returns
        -------
        self
        """
        return self.upd(x=x)

    def updy(self, y: float) -> 'Point':
        """
        Updates point y-coordinate.

        Parameters
        ----------
        y: float

        Returns
        -------
        self
        """
        return self.upd(y=y)

    def xy(self) -> Tuple[float, float]:
        return self.x, self.y


class MlObj(dict):
    """
    Node or edge object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_min_max(nodes: Dict[str, MlObj],
                edges: Dict[Tuple[str, str], MlObj]) -> Tuple[Point, Point]:
    """
    Returns the most lower-left and most upper-write points that
    depict the rectangular borders of the graph.

    Parameters
    ----------
    nodes: dict
        MlNodes.

    edges: dict
        MlEdges.

    Returns
    -------
    min_point, max_point: Point

    """
    x_arr = [obj['coords'].x for obj in nodes.values()]
    y_arr = [obj['coords'].y for obj in nodes.values()]
    for obj in edges.values():
        for p in obj['coords']:
            x_arr.append(p.x)
            y_arr.append(p.y)

    return Point(min(x_arr), min(y_arr)), Point(max(x_arr), max(y_arr))


def fix_pos(g: grand_graph.Graph,
            nodes_pos: Dict[str, Point],
            edges_pos: Dict[Tuple[str, str], List[Point]]) -> Tuple[Dict[str, Point],
                                                                    Dict[Tuple[str, str], List[Point]]]:
    """
    If graph consists of several subgraphs (connected components)
    place them above each other. The biggest subgraph (in terms of number
    of nodes) will be placed at the bottom, the smallest one at the top.

    Parameters
    ----------
    g: grand_graph.Graph
        Graph object from grandalf library.

    nodes_pos: dict
        Coordinates of nodes.

    edges_pos: dict
        Coordinates of edges.

    Returns
    -------
    nodes_pos: dict
        New coordinates of nodes.

    edges_pos: dict
        New coordinates of edges.
    """
    if len(g.C) == 1:
        return nodes_pos, edges_pos

    comps = sorted(g.C[:], key=lambda x: len(x.sV), reverse=True)

    ys = set([nodes_pos[v.view.data].y for v in comps[0].sV])
    if len(ys) == 1:  # 1 horizontal layer
        dist = max([v.view.h for v in comps[0].sV]) * 2  # height * 2
    else:
        dist = (max(ys) - min(ys)) / (len(ys) - 1)  # mean distance btw layers
    max_prev_y = max(ys)

    for comp in comps[1:]:
        miny = min([nodes_pos[v.view.data].y for v in comp.sV])
        maxy = max([nodes_pos[v.view.data].y for v in comp.sV])
        new_miny = max_prev_y + dist
        delta = new_miny - miny

        for v in comp.sV:
            nodes_pos[v.view.data].y += delta
        for e in comp.sE:
            v1, v2 = e.v
            edges_pos[(v1.data, v2.data)] = [p.updy(p.y + delta) for p in edges_pos[(v1.data, v2.data)]]

        max_prev_y = maxy + delta

    return nodes_pos, edges_pos


def get_pos(g: grand_graph.Graph) -> Tuple[Dict[str, Point],
                                           Dict[Tuple[str, str], List[Point]]]:
    """
    Get coordinates of nodes and edges from the graph object.

    Parameters
    ----------
    g: grand_graph.Graph
        Graph object from grandalf library.

    Returns
    -------
    nodes_pos: dict
        Coordinates of nodes.

    edges_pos: dict
        Coordinates of edges.
    """
    # coordinates of nodes
    nodes_coords = {}
    for str_con_comp in g.C:
        for v in str_con_comp.sV:
            nodes_coords[str(v.data)] = Point(v.view.xy[0], -v.view.xy[1])  # graph is upside down

    # coordinates of edges
    edges_coords = {}
    for str_con_comp in g.C:
        for e in str_con_comp.sE:
            v1, v2 = e.v
            points_list = [Point(x, -y) for (x, y) in e.view._pts]
            edges_coords[(str(v1.data), str(v2.data))] = points_list

    return fix_pos(g, nodes_coords, edges_coords)


def get_layout(graph: Union[Graph, ProcessTreeNode],
               iterations: int,
               rounded_corners: bool) -> Tuple[Dict[str, MlObj],
                                               Dict[Tuple[str, str], MlObj]]:
    """
    Get layout of the graph.

    Parameters
    ----------
    graph: Graph or ProcessTreeNode

    iterations: int
        Number of iterations done in the Sugiyama algorithm.

    rounded_corners: bool
        If True, makes the corners of edges round
        (but can be time consuming).

    Returns
    -------
    nodes: dict
        Nodes.

    edges: dict
        Edges.
    """
    nodes_dict = get_nodes(graph)
    edges_dict = get_edges(graph)
    V = {str(node): grand_graph.Vertex(str(node)) for node in nodes_dict.keys()}
    E = [grand_graph.Edge(V[str(n1)], V[str(n2)]) for (n1, n2) in edges_dict.keys()]
    V = list(V.values())

    g = grand_graph.Graph(V, E)
    k = 5
    for v in V:
        if nodes_dict[v.data].type == NodeType.TASK or \
                nodes_dict[v.data] == ProcessTreeNodeType.SINGLE_ACTIVITY and nodes_dict[v.data].label is not None:
            lines = v.data.split('\n')
            h = len(lines) * k
            w = max([len(line) for line in lines]) * 0.8 * k
        else:
            h = 1 * k
            w = h
        v.view = VertexViewer(data=v.data, w=w, h=h)
    for e in E:
        e.view = EdgeViewer()

    # algo
    for comp in g.C:
        sug = SugiyamaLayout(comp)
        if type(graph) == Graph:
            sug.xspace, sug.yspace = 5, 40
        else:
            sug.xspace, sug.yspace = 5, 20
        sug.dw, sug.dh = 1, 1
        sug.init_all(optimize=False)
        sug.draw(iterations + 0.5)

    # rounded corners
    if rounded_corners:
        for e in E:
            route_with_rounded_corners(e)

    # create objects
    nodes = {}
    for node_name, n in nodes_dict.items():
        obj = MlObj(dict(id=n.id, type=n.type, label=n.label))
        nodes[node_name] = obj

    edges = {}
    for (name1, name2), e in edges_dict.items():
        if type(graph) == Graph:
            obj = MlObj(dict(id=e.id, label=e.label))
        elif type(graph) == ProcessTreeNode:
            obj = MlObj(dict(id=str((name1, name2)), label=None))
        else:
            raise RuntimeError()
        edges[(name1, name2)] = obj

    nodes_pos, edges_pos = get_pos(g)

    # set coords
    for node_name, point in nodes_pos.items():
        nodes[node_name]['coords'] = point

    for edge_name, point_list in edges_pos.items():
        edges[edge_name]['coords'] = point_list

    # set left lower corner to (0, 0)
    set_lower_left_corner_zero(nodes, edges)

    return nodes, edges


def set_lower_left_corner_zero(nodes: Dict[str, MlObj],
                               edges: Dict[Tuple[str, str], MlObj]):
    pmin, _ = get_min_max(nodes, edges)
    for node in nodes.values():
        node['coords'] = node['coords'].add(pmin.neg())
    for edge in edges.values():
        edge['coords'] = list(map(lambda p: p.add(pmin.neg()), edge['coords']))


def draw_nodes(fig, ax,
               nodes: Dict[str, MlObj],
               fontsize: Optional[float] = 12) -> None:
    """
    Draws nodes and saves their coordinates to the node objects.

    Parameters
    ----------
    fig
    ax
    nodes
    fontsize
    """
    pad = 0.0
    for name, obj in nodes.items():
        node_type = obj['type']
        p = obj['coords']
        params = dict(x=p.x, y=p.y, color='black', fontsize=fontsize, ha='center', va='center')
        bbox_params = dict(facecolor='none', edgecolor='black', fill=True, boxstyle=f'round,pad={pad}')
        if node_type == NodeType.START_EVENT:
            params['s'] = '  \n  '
            bbox_params['facecolor'] = 'green'
            bbox_params['boxstyle'] = f'circle,pad={pad}'
        elif node_type == NodeType.END_EVENT:
            params['s'] = '  \n  '
            bbox_params['facecolor'] = 'red'
            bbox_params['boxstyle'] = f'circle,pad={pad}'
        elif node_type == NodeType.PLACE:
            params['s'] = '  \n  '
            bbox_params['boxstyle'] = f'circle,pad={pad}'
        elif node_type in NodeType.PARALLEL_GATEWAY:
            params['s'] = '+'
            bbox_params = dict(facecolor='none', edgecolor='none', boxstyle=f'round,pad={pad}')
        elif node_type == NodeType.EXCLUSIVE_GATEWAY:
            params['s'] = 'X'
            bbox_params = dict(facecolor='none', edgecolor='none', boxstyle=f'round,pad={pad}')
        elif node_type == NodeType.TASK:
            params['s'] = obj['label']
            if 'color' in obj:
                bbox_params['facecolor'] = obj['color']
        elif node_type == ProcessTreeNodeType.SINGLE_ACTIVITY and obj['label'] is not None:
            params['s'] = obj['label']
        elif node_type == ProcessTreeNodeType.SINGLE_ACTIVITY and obj['label'] is None:
            params['s'] = 'ABC'
            bbox_params['facecolor'] = 'black'
        elif node_type in [ProcessTreeNodeType.LOOP, ProcessTreeNodeType.PARALLEL, ProcessTreeNodeType.FLOWER,
                           ProcessTreeNodeType.SEQUENTIAL, ProcessTreeNodeType.EXCLUSIVE_CHOICE]:
            d = {ProcessTreeNodeType.LOOP: ' * ',
                 ProcessTreeNodeType.PARALLEL: '||',
                 ProcessTreeNodeType.FLOWER: ' ? ',
                 ProcessTreeNodeType.SEQUENTIAL: '->',
                 ProcessTreeNodeType.EXCLUSIVE_CHOICE: ' X '}
            params['s'] = f'\n{d[node_type]}\n'
            bbox_params['boxstyle'] = f'circle,pad={pad}'
        else:
            raise RuntimeError()
        params['bbox'] = bbox_params
        obj['text'] = ax.text(**params)

    fig.canvas.draw()

    # calc_bbox_coords
    for node_name, obj in nodes.items():
        we = obj['text'].get_window_extent()
        xmin, xmax, ymin, ymax = we.xmin, we.xmax, we.ymin, we.ymax

        xmin, ymin = ax.transData.inverted().transform((xmin, ymin))
        xmax, ymax = ax.transData.inverted().transform((xmax, ymax))

        obj['bbox_coords'] = [Point(xmin, ymin), Point(xmax, ymax)]

        if obj['type'] in [NodeType.PARALLEL_GATEWAY, NodeType.EXCLUSIVE_GATEWAY]:
            import matplotlib.patches as patches
            length = max([xmax - xmin, ymax - ymin])
            x, y = obj['coords'].xy()
            rect = patches.Rectangle((x, y - (length * 2 ** 0.5) / 2), length, length, 45,
                                     ec='black', fc='none')
            obj['bbox_coords'] = [Point(x - (length * 2 ** 0.5) / 2, y - (length * 2 ** 0.5) / 2),
                                  Point(x + (length * 2 ** 0.5) / 2, y + (length * 2 ** 0.5) / 2)]
            ax.add_patch(rect)


def plot_test_scatter(ax, nodes):
    for obj in nodes.values():
        pmin, pmax = obj['bbox_coords']
        x1, y1 = (ax.transAxes + ax.transData.inverted()).inverted().transform((pmin.x, pmin.y))
        ax.scatter(x1, y1, transform=ax.transAxes)
        ax.scatter(pmax.x, pmax.y, transform=ax.transData)


def get_cross_coords(node_obj: MlObj, p1: Point, p2: Point) -> Point:
    """
    Get coordinates of the point where the edge crosses the node box.

    Parameters
    ----------
    node_obj
    p1: Point
        Source point of the edge.
    p2: Point
        Target point of the edge.

    Returns
    -------
    result: Point
        Crossing point.
    """
    x1, y1 = p1.xy()
    x2, y2 = p2.xy()
    pmin, pmax = node_obj['bbox_coords']
    xmin, ymin = pmin.xy()
    xmax, ymax = pmax.xy()
    if node_obj['type'] in [NodeType.TASK, ProcessTreeNodeType.SINGLE_ACTIVITY]:
        if x1 == x2:
            cross = 'u' if max([y2, y1]) > ymax else 'd'
            x_cross = x1
            y_cross = ymax if cross == 'u' else ymin
        else:
            tg_node = (ymax - ymin) / (xmax - xmin)
            tg_line = (y2 - y1) / (x2 - x1)
            if -tg_node < tg_line < tg_node:  # ud
                cross = 'r' if max([x2, x1]) > xmax else 'l'
            else:
                cross = 'u' if max([y2, y1]) > ymax else 'd'

            k = tg_line
            b = y1 - k * x1
            if cross in ['u', 'd']:
                y_cross = ymax if cross == 'u' else ymin
                x_cross = (y_cross - b) / k
            else:
                x_cross = xmax if cross == 'r' else xmin
                y_cross = k * x_cross + b

    elif node_obj['type'] in [NodeType.PLACE, NodeType.START_EVENT, NodeType.END_EVENT] + [
        ProcessTreeNodeType.LOOP, ProcessTreeNodeType.FLOWER, ProcessTreeNodeType.PARALLEL,
        ProcessTreeNodeType.SEQUENTIAL, ProcessTreeNodeType.EXCLUSIVE_CHOICE]:
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        r2 = ((xmax - xc) ** 2 + (ymax - yc) ** 2)

        if x1 == x2:
            x_cross = x1
            y_cross_lower = yc - r2 ** 0.5
            y_cross = yc + r2 ** 0.5 if y_cross_lower < min([y1, y2]) else y_cross_lower
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            a_ = k * k + 1
            b_ = 2 * (k * b - k * yc - xc)
            c_ = xc ** 2 + b ** 2 + yc ** 2 - 2 * b * yc - r2
            D = b_ ** 2 - 4 * a_ * c_
            x1_ = (-b_ + D ** 0.5) / (2 * a_)
            x2_ = (-b_ - D ** 0.5) / (2 * a_)

            x_cross = x1_ if min([x1, x2]) < x1_ < max([x1, x2]) else x2_
            y_cross = k * x_cross + b
    elif node_obj['type'] in [NodeType.PARALLEL_GATEWAY, NodeType.EXCLUSIVE_GATEWAY]:
        r = (xmax - xmin) / 2  # 2*r = diagonal
        if xmin < p1.x < xmax and ymin < p1.y < ymax:
            pin = p1
            pout = p2
        else:
            pin = p2
            pout = p1
        if pout.x == pin.x:
            x_cross = pout.x
            if pout.y > pin.y:
                y_cross = pin.y + r
            else:
                y_cross = pin.y - r
        else:
            k = (pout.y - pin.y) / (pout.x - pin.x)
            if -1 <= k <= 1:
                if pout.x > pin.x:
                    x_cross, y_cross = pin.x + r, pin.y  # right
                else:
                    x_cross, y_cross = pin.x - r, pin.y  # left
            else:
                if pout.y > pin.y:
                    x_cross, y_cross = pin.x, pin.y + r  # up
                else:
                    x_cross, y_cross = pin.x, pin.y - r  # down

    else:
        raise RuntimeError(f"type {node_obj['type']} not fount")

    return Point(x_cross, y_cross)


class MlPainter:

    def __init__(self):
        self.fig = None
        self.ax = None
        self.nodes, self.edges = None, None

    def apply(self, graph: Union[Graph, ProcessTreeNode],
              node_style_metric=None,
              edge_style_metric=None,
              hide_disconnected_nodes: Optional[bool] = False,
              iterations: Optional[int] = 5,
              rounded_corners: Optional[bool] = False,
              vertical: Optional[bool] = True,
              figsize: Optional[int] = 10,
              fontsize: Optional[int] = 10):
        """

        Parameters
        ----------
        graph : Graph or ProcessTreeNode
            Graph object.

        node_style_metric: str
            Name of the node's metric that will influence the colour of the nodes.
            If None or given metric in not contained in the given graph, nodes will have the same colour.
            Is not used if graph is a ProcessTreeNode object.

        edge_style_metric: str
            Name of the edge's metric that will influence the thickness of the edges.
            If None or given metric in not contained in the given graph, edges will have the same width.
            Is not used if graph is a ProcessTreeNode object.

        hide_disconnected_nodes: bool, default=True
            If True, nodes without any input and output edges will not be displayed.
            Is not used if graph is a ProcessTreeNode object.

        iterations: int
            Number of iterations done in the Sugiyama algorithm.

        rounded_corners: bool
            If True, makes the corners of edges round
            (but can be time consuming).

        vertical: bool, default=True.
            If True, graph is drawn from top to bottom, otherwise from left to right.

        figsize
        fontsize
        """
        with plt.ioff():  # so that the picture is not shown after the method call
            # clear figure and axes
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            self.fig, self.ax = fig, ax
            ax.axis('scaled')

            if hide_disconnected_nodes and type(graph) == Graph:
                graph = copy.deepcopy(graph)
                nodes_to_exclude = set([node.id for node in graph.nodes.values()
                                        if len(node.input_edges) == len(node.output_edges) == 0])
                for n in nodes_to_exclude:
                    graph.remove_node_by_id(n)

            # get coords from algo
            nodes, edges = get_layout(graph, iterations=iterations, rounded_corners=rounded_corners)
            self.nodes, self.edges = nodes, edges

            if vertical is False:
                make_graph_horizontal(nodes, edges)

            # set axes limits
            p_min, p_max = get_min_max(nodes, edges)
            # +-10 so that all the lines are inside
            ax.set_xlim(p_min.x - 10, p_max.x + 10)
            ax.set_ylim(p_min.y - 10, p_max.y + 10)
            max_ax_lim = max(p_max.x - p_min.x, p_max.y - p_min.y)

            # include metrics info into nodes' names
            if type(graph) == Graph:
                add_metrics_to_node_names(graph, nodes)

            # nodes colors
            if node_style_metric is not None and type(graph) == Graph:
                add_nodes_colors(graph, nodes, node_style_metric)
            if edge_style_metric is not None and type(graph) == Graph:
                add_edges_widths(graph, edges, edge_style_metric)

            # draw nodes and get coords of the surrounding boxes
            draw_nodes(fig, ax, nodes, fontsize=fontsize)

            # change first and last edges' coords (after nodes are drawn!)
            for (n1, n2), obj in edges.items():
                # first edge
                p1 = obj['coords'][0]
                p2 = obj['coords'][1]
                p_cross = get_cross_coords(nodes[n1], p1, p2)
                obj['coords'][0] = p_cross

                # last edge
                p1 = obj['coords'][-2]
                p2 = obj['coords'][-1]
                p_cross = get_cross_coords(nodes[n2], p1, p2)
                obj['coords'][-1] = p_cross

            # if axis_option == 'auto':
            #     transformer = ax.transAxes
            #     transform_edge_data_to_axis(ax, edges)
            # else:
            #     transformer = ax.transData
            transformer = ax.transData

            # set edge_width coeff

            edge_width_coeff = (1 / 750 * max_ax_lim) ** 1.5 + 0.5
            edge_width_coeff = min([2, edge_width_coeff])
            # draw edges
            for obj in edges.values():
                # plot all the edges except the last one
                # the first parts of the edge are drawn as edges without the head
                # so that their width is the same as one of the last one.
                width = obj['width'] if 'width' in obj else 0.8
                width *= edge_width_coeff
                if len(obj['coords']) >= 3:
                    for i in range(len(obj['coords']) - 2):
                        x1, y1 = obj['coords'][i].xy()
                        x2, y2 = obj['coords'][i + 1].xy()
                        ax.arrow(x=x1, y=y1, dx=x2 - x1, dy=y2 - y1,
                                 length_includes_head=True,
                                 fc='black', transform=transformer, linewidth=0,
                                 width=width, head_width=0.0, head_length=0.0)

                # the last part of the edge
                x1, y1 = obj['coords'][-2].xy()
                x2, y2 = obj['coords'][-1].xy()

                _ = ax.arrow(x=x1, y=y1, dx=x2 - x1, dy=y2 - y1,
                             length_includes_head=True,
                             fc='black', transform=transformer, linewidth=0,
                             width=width, head_width=3 * edge_width_coeff, head_length=4.5 * edge_width_coeff)

    def show(self) -> plt.Figure:
        """
        Shows visualization of the graph in Jupyter Notebook.

        Returns
        -------
        digraph : IPython.core.display.HTML
            Graph in HTML format.
        """
        return self.fig

    def save(self, filename, dpi=None, **kwargs):
        """
        Saves a graph visualization to file.

        Parameters
        ----------
        filename : str
            Name of the file to save the result to.

        dpi: float or 'figure', default: :rc:`savefig.dpi`
            The resolution in dots per inch.  If 'figure', use the figure's
            dpi value.

        kwargs:
            Additional maplotlib.pyplot.savefig() parameters
            (excluding 'fname' and 'dpi').
        """
        self.fig.savefig(fname=filename, dpi=dpi, **kwargs)

    def write_graph(self, filename, dpi=None, **kwargs):
        """
        Saves a graph visualization to file.

        Parameters
        ----------
        filename : str
            Name of the file to save the result to.

        dpi: float or 'figure', default: :rc:`savefig.dpi`
            The resolution in dots per inch.  If 'figure', use the figure's
            dpi value.
        """
        self.save(filename, dpi, **kwargs)


# def transform_edge_data_to_axis(ax, edges):
#     axis_to_data = ax.transAxes + ax.transData.inverted()
#     data_to_axis = axis_to_data.inverted()
#
#     for obj in edges.values():
#         obj['coords'] = [data_to_axis.transform((x, y)) for (x, y) in obj['coords']]
#


def add_nodes_colors(graph, nodes, node_style_metric: str):
    colors_dict = utils.calc_nodes_colors_by_metric(graph, node_style_metric)
    for name, obj in nodes.items():
        obj['color'] = colors_dict[name]


def add_edges_widths(graph, edges, edge_style_metric: str):
    widths_dict = utils.calc_edges_widths_by_metric(graph, edge_style_metric, 0.3, 2)
    for name, obj in edges.items():
        if len({graph.nodes[name[0]].type, graph.nodes[name[1]].type}
                       .intersection([NodeType.START_EVENT, NodeType.END_EVENT])) == 0:
            obj['width'] = widths_dict[name]


def add_metrics_to_node_names(graph: Graph, nodes):
    for node_name, obj in nodes.items():
        label = utils.add_metrics_to_node_label(obj['label'],
                                                graph.nodes[node_name].metrics,
                                                newline='\n')
        obj['label'] = label


def get_nodes(graph) -> Dict[str, Union[ProcessTreeNode, Node]]:
    if type(graph) == Graph:
        return graph.nodes
    elif type(graph) == ProcessTreeNode:
        return graph.get_nodes()


def get_edges(graph) -> Dict[Tuple[str, str], Union[Tuple[ProcessTreeNode, ProcessTreeNode], Edge]]:
    if type(graph) == Graph:
        return graph.edges
    elif type(graph) == ProcessTreeNode:
        return graph.get_edges()


def make_graph_horizontal(nodes: Dict[str, MlObj],
                          edges: Dict[Tuple[str, str], MlObj]):
    for node in nodes.values():
        node['coords'] = Point(-node['coords'].y, node['coords'].x)

    for edge in edges.values():
        edge['coords'] = [Point(-p.y, p.x) for p in edge['coords']]

    set_lower_left_corner_zero(nodes, edges)
