from ._bpmn_xml_maker import XMLMaker
from ._petri_net_to_bpmn import petri_net_to_bpmn
from .grandalf import calc_coords_grandalf
from .graphviz import calc_coords_graphviz
from ...visual._graph import Graph
from ...visual._types import GraphType


class BpmnExporter:
    """
    Converts a Petri net to BPMN graph and saves it to a .bpmn-file.

    Attributes
    ----------
    graph: Graph
        BPMN or DFG graph that will be transformed into xml format.

    _xml_maker: XMLMaker
        Object that transforms a bpmn graph to an xml representation.

    Examples
    --------
    >>> from sberpm.bpmn import BpmnExporter
    >>>
    >>> bpmn_exporter = BpmnExporter()
    >>> bpmn_exporter.apply(graph)
    >>> bpmn_exporter.save('file_name.bpmn')
    >>> print(bpmn_exporter.get_string_representation())
    """

    def __init__(self):
        self._xml_maker = None
        self.graph = None

    def apply(self, graph: Graph, engine: str = None):
        """
        Converts a given graph to BPMN notation (xml).

        Parameters
        ----------
        graph : Graph
            Graph object.
            In case of Petri-Net, the graph will be converted to BPMN-graph first.
            In case of DFG, it will not be converted to BPMN-graph but
            will be directly transformed into XML-notation.

            * As DFG graph has no information about types of gateways,
            the resulting BPMN representation will be without gateways,
            making it NOT a valid BPMN notation.

        engine: {'graphviz', 'grandalf'}, default=None
            Program used to calculate coordinates.
            If None, 'graphviz' will be used if it is installed, 'grandalf' otherwise.
        """
        if graph.type == GraphType.PETRI_NET:
            graph = petri_net_to_bpmn(graph)
        self.graph = graph
        self._apply(graph, engine)

    def _apply(self, graph, engine=None):

        # Engine check
        if engine not in ['graphviz', 'grandalf', None]:
            raise TypeError(f"engine must be one of 'graphviz', 'grandalf' or None, "
                            f'but got {engine}.')
        if engine is None:
            engine = 'graphviz' if graphviz_is_installed() else 'grandalf'

        # Calculate coordinates and nodes' sizes.
        if engine == 'graphviz':
            node_params_dict, edge_pos_dict = calc_coords_graphviz(graph)
        else:
            node_params_dict, edge_pos_dict = calc_coords_grandalf(graph)

        # Transform graph to xml
        self._xml_maker = XMLMaker(graph, node_params_dict, edge_pos_dict)

    def write(self, filename: str):
        """
        Saves calculated BPMN graph in BPMN notation to a file.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        self.save(filename)

    def save(self, filename: str):
        """
        Saves calculated BPMN graph in BPMN notation to a file.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        if self._xml_maker is None:
            raise RuntimeError('Call apply() first.')
        else:
            self._xml_maker.save(filename)

    def get_string_representation(self) -> str:
        """
        Returns a string representation of BPMN notation of calculated BPMN graph.

        Returns
        -------
        result: str
            BPMN notation of the graph.
        """
        if self._xml_maker is None:
            raise RuntimeError('Call apply_petri() or apply_dfg() first.')
        else:
            return self._xml_maker.to_string()


def graphviz_is_installed() -> bool:
    """
    Check whether graphviz is installed.

    Returns
    -------
    result: bool
        True if installed, False otherwise.
    """
    import os
    return bool(1 - os.system('dot -V'))
