import xml.dom.minidom
import xml.etree.ElementTree as eTree
from io import BytesIO
from typing import Tuple, List, Dict, Union

from ...visual._graph import Graph
from ...visual._types import NodeType


class XMLMaker:
    """
    Transforms bpmn-graph to bpmn-file.

    Parameters
    ---------
    bpmn_graph: Graph
        Graph of type BPMN.

    node_params_dict: dict
        Parameters of nodes. Keys are the same as bpmn_graph.nodes.keys(),
        values include: 'label', 'height', 'width', 'x', 'y'.

    edge_pos_dict: dict
        Waypoints (coordinates) of edges. Keys are the same as bpmn_graph.edges.keys(),
        value is a list of tuples (x, y).
    """

    def __init__(self, bpmn_graph: Graph,
                 node_params_dict: Dict[str, Dict[str, Union[str, float]]],
                 edge_pos_dict: Dict[Tuple[str, str], List[Tuple[float, float]]]) -> None:
        self._root = None
        self.bpmn_graph = bpmn_graph
        self.node_params_dict = node_params_dict
        self.edge_pos_dict = edge_pos_dict
        self._do()

    def _do(self):
        def pref_di(s: str) -> str:
            return 'bpmndi' + ':' + s

        self._root = self._create_definitions()
        bpmn_graph = self.bpmn_graph
        node_params_dict = self.node_params_dict
        edge_pos_dict = self.edge_pos_dict

        # 1. Create objects
        process_id = 'Process_123456'
        process = eTree.SubElement(self._root, 'process', {"id": process_id})

        # create bpmn nodes and give them IDs
        bpmn_nodes_ids = {}
        bpmn_nodes_elements = {}
        counters = {}
        for name, graph_node in bpmn_graph.nodes.items():
            if graph_node.type == NodeType.START_EVENT:
                bpmn_type = 'startEvent'
            elif graph_node.type == NodeType.END_EVENT:
                bpmn_type = 'endEvent'
            elif graph_node.type == NodeType.TASK:
                bpmn_type = 'task'
            elif graph_node.type == NodeType.PARALLEL_GATEWAY:
                bpmn_type = 'parallelGateway'
            elif graph_node.type == NodeType.EXCLUSIVE_GATEWAY:
                bpmn_type = 'exclusiveGateway'
            else:
                raise ValueError(f'Got bpmn node of type "{graph_node.type}"')
            if bpmn_type not in counters:
                counters[bpmn_type] = 0
            obj_id = f'{bpmn_type}_{counters[bpmn_type]}'
            bpmn_nodes_ids[name] = obj_id
            counters[bpmn_type] += 1
            bpmn_nodes_elements[name] = eTree.SubElement(process, bpmn_type, {'id': obj_id})
            if graph_node.type == NodeType.TASK:
                bpmn_nodes_elements[name].set('name', graph_node.label)

        # create bpmn edges and give them IDs, modify bpmn nodes
        bpmn_edges_ids = {}
        edge_counter = 0
        bpmn_type = 'sequenceFlow'
        for (n1, n2), edge in bpmn_graph.edges.items():
            # for (source_name, dest_name), pos in edge_pos_dict.items():
            obj_id = f'{bpmn_type}_{edge_counter}'
            bpmn_edges_ids[(n1, n2)] = obj_id
            edge_counter += 1
            eTree.SubElement(process, bpmn_type,
                             {'id': obj_id, "sourceRef": bpmn_nodes_ids[n1], "targetRef": bpmn_nodes_ids[n2]})

            # add 'incoming' and 'outgoing' to bpmn nodes
            id1 = bpmn_nodes_ids[n1]
            id2 = bpmn_nodes_ids[n2]
            bpmn_node1 = bpmn_nodes_elements[n1]
            bpmn_node2 = bpmn_nodes_elements[n2]
            bpmn_node1.set("outgoing", id2)
            bpmn_node2.set("incoming", id1)

        # 2. Create diagram (coordinates and sizes).
        diagram = eTree.SubElement(self._root, pref_di('BPMNDiagram'), {"id": "Diagram_123456"})
        plane = eTree.SubElement(diagram, pref_di('BPMNPlane'), {
            "id": "Plane_123456",
            'bpmnElement': process_id
        })

        # nodes' coordinates
        for node, params in node_params_dict.items():
            node_id = bpmn_nodes_ids[node]
            bpmn_element = eTree.SubElement(plane, pref_di('BPMNShape'),
                                            {"id": f'element_{node_id}', "bpmnElement": node_id})
            eTree.SubElement(bpmn_element, 'omgdc:Bounds', {
                'x': str(params['x']),
                'y': str(params['y']),
                'width': str(params['width']),
                'height': str(params['height']),
            })

        # edges' coordinates
        for (n1, n2), xy_list in edge_pos_dict.items():
            edge_id = bpmn_edges_ids[(n1, n2)]

            bpmn_element = eTree.SubElement(plane, pref_di('BPMNEdge'))
            bpmn_element.set("id", f'element_{edge_id}')
            bpmn_element.set("bpmnElement", edge_id)
            for x, y in xy_list:
                eTree.SubElement(bpmn_element, 'omgdi:waypoint', {
                    "x": str(round(x, 3)),
                    "y": str(round(y, 3))
                })

    @staticmethod
    def _create_definitions() -> eTree.Element:
        """
        Set the beginning of the xml-file.

        Returns
        -------
        root: eTree.Element
            Root element of xml (with definitions).
        """
        root = eTree.Element('definitions')
        root.set("xmlns", "http://www.omg.org/spec/BPMN/20100524/MODEL")
        root.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
        root.set("xmlns:omgdi", "http://www.omg.org/spec/DD/20100524/DI")
        root.set("xmlns:omgdc", "http://www.omg.org/spec/DD/20100524/DC")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("targetNamespace", "http://bpmn.io/schema/bpmn")
        root.set("id", "Definitions_123456")
        return root

    def save(self, filename: str) -> None:
        """
        Save to bpmn-file.

        Parameters
        ----------
        filename: str
            File name or path (including file name).
        """
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write(self.to_string())

    def to_string(self) -> str:
        """
        Get string representation of bpmn graph.

        Returns
        -------
        str
        """
        tree = eTree.ElementTree(self._root)
        f = BytesIO()
        tree.write(f, encoding='utf-8', xml_declaration=True)
        str_xml = f.getvalue().decode('utf-8')
        dom = xml.dom.minidom.parseString(str_xml)
        pretty_xml_as_string = dom.toprettyxml()
        return pretty_xml_as_string
