from xml.dom import minidom

import pydotplus

from ..visual._graph import Graph, create_bpmn
from ..visual._types import NodeType


class LocalObject:
    """
    Abstract class. Represents an object imported from a bpmn-file.
    """

    def __init__(self, attrs, obj_type):
        self.attrs = attrs
        self.id = attrs['id']
        self.type = obj_type

    def has_attr(self, attr_name):
        return attr_name in self.attrs

    def get_attr(self, attr_name):
        return self.attrs[attr_name]


class LocalNode(LocalObject):
    """
    Represents a node imported from a bpmn-file.
    """

    def __init__(self, attrs, obj_type):
        super().__init__(attrs, obj_type)
        self.name = attrs['name'].replace('\n', '').replace('\r', '') if 'name' in attrs else ''
        # self.name = attrs['name'][:20] if 'name' in attrs else ''

        self.parent_node = None
        self.child_nodes = []

        self.incoming_edges = []
        self.outgoing_edges = []

        self.is_separate_node = True

    def set_parent(self, node: 'LocalNode'):
        self.parent_node = node

    def add_child_node(self, node: 'LocalNode'):
        self.child_nodes.append(node)
        self.is_separate_node = False

    def remove_child_node(self, node):
        self.child_nodes.remove(node)
        if len(self.child_nodes) == 0:
            self.is_separate_node = True

    def add_incoming_edge(self, edge: 'LocalEdge'):
        self.incoming_edges.append(edge)

    def add_outgoing_edge(self, edge: 'LocalEdge'):
        self.outgoing_edges.append(edge)

    def get_subgraph_objects(self):
        """
        Return inner objects that are not separate nodes
        """
        return [obj for obj in self.child_nodes if not obj.is_separate_node]

    def get_start_nodes(self):
        """
        Return 'startEvent' object of this object and of its inner objects
        """
        node_list = []
        for node in self.child_nodes:
            node_list += [node] if node.type == 'startEvent' else node.get_start_nodes()
        return node_list

    def get_end_nodes(self):
        """
        Return 'endEvent' object of this object and of its inner objects
        """
        node_list = []
        for node in self.child_nodes:
            node_list += [node] if node.type == 'endEvent' else node.get_end_nodes()
        return node_list


class LocalEdge(LocalObject):
    """
    Represents an edge imported from a bpmn-file.
    """

    def __init__(self, attrs: dict, edge_id=None, edge_type='sequenceFlow'):
        if len(attrs) != 0:
            super().__init__(attrs, edge_type)
        else:
            if edge_id is None:
                raise ValueError('If "attrs" is an empty dict, "edge_id" must be provided')
            self.id = edge_id
            self.type = edge_type
        self.name = attrs['name'] if 'name' in attrs else ''
        self.source_node = None
        self.target_node = None

        # Contains name of this edge and deleted edges that were substituted by this one
        self.object_names = [self.name] if self.name != '' else []

    def set_source_node(self, node: LocalNode):
        self.source_node = node
        return self

    def set_target_node(self, node: LocalNode):
        self.target_node = node
        return self

    def add_deleted_objects_names(self, edge_in: 'LocalEdge', node: 'LocalNode', edge_out: 'LocalEdge'):
        self.object_names = edge_in.get_object_names() + [node.name] + edge_out.get_object_names()
        return self

    def set_objects_names(self, object_names: list):
        self.object_names = object_names
        return self

    def get_object_names(self):
        return self.object_names


class DataContainer:
    """
    Represents a graph. Contains information about nodes and edges.
    """

    def __init__(self, additional_tags_to_ignore: (list, None)):
        self.all_objects = {}
        self.edges_source_target_dict = {}

        self.ignored_tags = ['incoming', 'outgoing', 'compensateEventDefinition', 'ioSpecification',
                             'messageEventDefinition', 'dataInputAssociation', 'timerEventDefinition']
        self.edges_tags = ['sequenceFlow', 'messageFlow']
        self.types_that_can_have_dependant_objects = ['process', 'subProcess']

        if additional_tags_to_ignore is not None:
            if isinstance(additional_tags_to_ignore, list):
                self.ignored_tags += additional_tags_to_ignore
                self.edges_tags = list(set(self.edges_tags) - set(self.ignored_tags))
            else:
                raise TypeError(f'tags_to_ignore must be of type list or None, but got: '
                                f'{type(additional_tags_to_ignore)}')

    def add_edge(self, edge):
        """
        Adds a LocalEdge() object to local data structures
        """
        source_node = edge.source_node
        target_node = edge.target_node
        source_node.add_outgoing_edge(edge)
        target_node.add_incoming_edge(edge)
        if source_node.id not in self.edges_source_target_dict:
            self.edges_source_target_dict[source_node.id] = {}
        self.edges_source_target_dict[source_node.id][target_node.id] = edge
        self.all_objects[edge.id] = edge

    def remove_edge(self, in_node, out_node):
        """
        Removes a LocalEdge() object from local data structures
        """
        edge = self.edges_source_target_dict[in_node.id][out_node.id]
        del self.edges_source_target_dict[in_node.id][out_node.id]
        del self.all_objects[edge.id]
        if len(self.edges_source_target_dict[in_node.id].keys()) == 0:
            del self.edges_source_target_dict[in_node.id]
        in_node.outgoing_edges.remove(edge)
        out_node.incoming_edges.remove(edge)

    def remove_node(self, node):
        """
        Removes a LocalNode() object from local data structures
        """
        if node.parent_node is not None:
            node.parent_node.child_nodes.remove(node)
        del self.all_objects[node.id]


class BpmnImporter:
    """
    Loads .xml/.bpmn file that contains a graph in a bpmn-notation and visualizes it.

    Attributes
    ----------
    _data: DataContainer
        Object that contains a graph read from the file.

    Examples
    --------
    >>> from sberpm.bpmn import BpmnImporter
    >>>
    >>> bpmn_importer = BpmnImporter().load_bpmn_from_xml('file.bpmn')
    >>> pydot_graph = bpmn_importer.get_pydotplus_graph()
    >>> pydot_graph.write('imported_bpmn.svg', prog='dot', format='svg')
    """

    def __init__(self):
        self._data = None

    def load_bpmn_from_xml(self, file_path, additional_tags_to_ignore: (list, None) = None, remove_gateways=False):
        """
        Reads an XML file from given file_path and maps it into inner representation of BPMN diagram.

        Parameters
        ----------
        file_path: str
            Path to the .bpmn/.xml file.

        additional_tags_to_ignore: list of str or None, default=None
            Types of objects of a bpmn graph that need to be ignored from reading.

        remove_gateways: bool, default=False
            If True, gateways are removed and the incoming and outgoing nodes are reconnected.
        """
        self._data = DataContainer(additional_tags_to_ignore)

        document = BpmnImporter._read_xml_file(file_path)

        # Import process elements and their dependant objects (except edges)
        for process_element in document.getElementsByTagNameNS("*", 'process'):
            self._import_nodes_and_dependant_nodes(process_element, None)

        # Import edges
        for flow_type in self._data.edges_tags:
            for edge_element in document.getElementsByTagNameNS("*", flow_type):
                self._import_edge(edge_element)

        # Make nodes, that have 'calledElement' references, parent nodes for those called elements
        self._make_call_elements_children()

        # Make edges if a node has an 'attachedToRef' attribute
        self._make_edges_for_references()

        # Remove Gateways
        if remove_gateways:
            self._remove_gateways()

        # Substitute a source or a target object of the edge with a real node if it is a 'process' node
        self._ensure_all_connections_are_between_separate_nodes()

        # Remove disconnected nodes
        for obj in list(self._data.all_objects.values()):
            if isinstance(obj, LocalNode):
                if obj.is_separate_node and len(obj.incoming_edges) == 0 and len(obj.outgoing_edges) == 0:
                    del self._data.all_objects[obj.id]

        return self

    @staticmethod
    def _read_xml_file(file_path):
        """
        Reads BPMN 2.0 XML file.

        Parameters
        ----------
        file_path: str
            Path to the .bpmn/.xml file.

        Returns
        -------
        dom_tree: xml.dom.xminidom.Document
            XML representation of the file.
        """
        dom_tree = minidom.parse(file_path)
        return dom_tree

    def _import_nodes_and_dependant_nodes(self, element: minidom.Element, parent_node: (LocalNode, None)):
        """
        Imports an object element from bpmn (creates a LocalNode object).
        If this object is supposed to have children (''process' or 'subProcess'), calls this function recursively.

        Parameters
        ----------
        element: minidom.Element
            Element that needs to be imported.
        parent_node: LocalNode or None
            Parental node of the given element.
        """
        tag_name = BpmnImporter._remove_namespace_from_tag_name(element.tagName)

        if tag_name in self._data.ignored_tags:
            return
        if tag_name in self._data.edges_tags:
            return

        attrs = BpmnImporter._get_attributes(element)
        if 'id' not in attrs:  # if an object does not have an 'id', it is strange
            return
        node = LocalNode(attrs, tag_name)
        self._data.all_objects[node.id] = node

        if parent_node is not None:
            parent_node.add_child_node(node)
            node.set_parent(parent_node)

        if tag_name in self._data.types_that_can_have_dependant_objects:
            for child_element in BpmnImporter._iterate_elements(element):
                if child_element.nodeType != child_element.TEXT_NODE:
                    self._import_nodes_and_dependant_nodes(child_element, node)

    def _import_edge(self, edge_element: minidom.Element):
        """
        Imports a flow element from bpmn (creates a LocalEdge object).

        Parameters
        ----------
        edge_element: minidom.Element
            Element that needs to be imported.
        """
        attrs = BpmnImporter._get_attributes(edge_element)
        source_ref = attrs['sourceRef']
        target_ref = attrs['targetRef']
        if source_ref in self._data.all_objects.keys() and target_ref in self._data.all_objects.keys():
            source_node = self._data.all_objects[source_ref]
            target_node = self._data.all_objects[target_ref]
            edge = LocalEdge(attrs) \
                .set_source_node(source_node) \
                .set_target_node(target_node)
            self._data.add_edge(edge)

    def _make_call_elements_children(self):
        """
        If an object's type is 'callActivity' and it has a 'calledElement' reference,
        this object becomes a parent node for a 'calledElement' node.
        """
        for obj in self._data.all_objects.values():
            if isinstance(obj, LocalNode):
                if obj.type == 'callActivity' and obj.has_attr('calledElement'):
                    called_element_id = obj.get_attr('calledElement')
                    if called_element_id in self._data.all_objects:
                        called_element = self._data.all_objects[called_element_id]
                        obj.add_child_node(called_element)
                        called_element.set_parent(obj)

    def _make_edges_for_references(self):
        """
        If a node has 'attachedToRef' attribute, makes an edge
        from  'attachedToRef' node to this node.
        """
        for obj in list(self._data.all_objects.values()):
            if obj.has_attr('attachedToRef'):
                out_node = obj
                obj_1 = self._data.all_objects[out_node.get_attr('attachedToRef')]
                source_node_list = [self._data.all_objects[obj_1.id]] if obj_1.is_separate_node else \
                    [self._data.all_objects[last_node.id] for last_node in obj_1.get_end_nodes()]
                target_node_list = [self._data.all_objects[out_node.id]] if out_node.is_separate_node else \
                    [self._data.all_objects[first_node.id] for first_node in out_node.get_start_nodes()]
                for n1 in source_node_list:
                    for n2 in target_node_list:
                        edge = LocalEdge({}, n1.id + n2.id, 'attachedToRef') \
                            .set_source_node(n1) \
                            .set_target_node(n2)
                        self._data.add_edge(edge)

    def _remove_gateways(self):
        """
        Removes gateways from the graph.
        """
        for obj in list(self._data.all_objects.values()):
            if 'Gateway' in obj.type:
                gnode = obj
                in_nodes = [e.source_node for e in gnode.incoming_edges]
                out_nodes = [e.target_node for e in gnode.outgoing_edges]
                for in_node in in_nodes:
                    for out_node in out_nodes:
                        # If edge does not exist, make one
                        if not (in_node.id in self._data.edges_source_target_dict and
                                out_node.id in self._data.edges_source_target_dict[in_node.id]):
                            in_edge = self._data.edges_source_target_dict[in_node.id][gnode.id]
                            out_edge = self._data.edges_source_target_dict[gnode.id][out_node.id]
                            new_edge = LocalEdge({}, edge_id=in_edge.id + out_edge.id) \
                                .set_source_node(in_node) \
                                .set_target_node(out_node) \
                                .add_deleted_objects_names(in_edge, gnode, out_edge)
                            self._data.add_edge(new_edge)
                # Remove old edges and the gateway node
                for in_node in in_nodes:
                    self._data.remove_edge(in_node, gnode)
                for out_node in out_nodes:
                    self._data.remove_edge(gnode, out_node)
                self._data.remove_node(gnode)

    def _ensure_all_connections_are_between_separate_nodes(self):
        """
        If f.e. a source object of the edge is a node that has children,
        we suppose that this node should not be displayed in the graph.
        The connection between this node and target node is substituted
        with connections from its inner nodes (children) and the target node.
        """
        for obj in list(self._data.all_objects.values()):
            if isinstance(obj, LocalEdge):
                edge = obj
                source_is_separate = edge.source_node.is_separate_node
                target_is_separate = edge.target_node.is_separate_node
                if source_is_separate and target_is_separate:
                    continue
                source_nodes = [edge.source_node] if source_is_separate else edge.source_node.get_end_nodes()
                target_nodes = [edge.target_node] if target_is_separate else edge.target_node.get_start_nodes()

                self._data.remove_edge(edge.source_node, edge.target_node)
                for source_node in source_nodes:
                    for target_node in target_nodes:
                        new_edge = LocalEdge({}, edge_id=source_node.id + target_node.id) \
                            .set_source_node(source_node) \
                            .set_target_node(target_node) \
                            .set_objects_names(edge.object_names)
                        self._data.add_edge(new_edge)

    @staticmethod
    def _get_attributes(element: minidom.Element):
        """
        Returns attributes of an element.

        Parameters
        ----------
        element: minidom.Element
            Given element.

        Returns
        -------
        attrs: dict
            Attributes. Key: attribute's name, value: its value.

        """
        element._ensure_attributes()
        attrs = {a: element.getAttribute(a) for a in element._attrs}
        return attrs

    @staticmethod
    def _remove_namespace_from_tag_name(tag_name: str):
        """
        Removes namespace annotation from tag name (f.e., semantic:startEvent -> startEvent).

        Parameters
        ----------
        tag_name: str
            Full tag's name.

        Returns
        -------
        result: str
            The tag's name without namespace.
        """
        return tag_name.split(':')[-1]

    @staticmethod
    def _iterate_elements(element: minidom.Element):
        """
        Iterates over child Nodes/Elements of parent Node/Element.

        Parameters
        ----------
        element: minidom.Element
            Given element.

        Returns
        -------
        elements: generator of minidom.Element
            Child elements of the given element.
        """
        element = element.firstChild
        while element is not None:
            yield element
            element = element.nextSibling

    def get_pydotplus_graph(self, show_edge_labels=False, vertical=True):
        """
        Returns the visualization object of the imported bpmn-graph.

        Parameters
        ----------
        show_edge_labels: boolean, default=False
            If True, edges' labels are displayed in the graph.

        vertical: boolean, default=True
            If True, the direction from the first to the last nodes goes from up to down.

        Returns
        -------
        pydotplus_graph: pydotplus.Dot
            Graph that can be visualized.
        """
        pydotplus_graph = PydotPlusGraphMaker.make_graph(self._data, show_edge_labels=show_edge_labels,
                                                         vertical=vertical, orthogonal_lines=False)

        return pydotplus_graph

    def get_bpmn_graph(self) -> Graph:
        """
        Returns the graph of type BPMN.

        Returns
        -------
        graph: Graph
            Graph that can be visualized.
        """
        graph = BpmnGraphMaker.make_graph(self._data)

        return graph


class PydotPlusGraphMaker:
    """
    Class that uses a pydotplus package (that has graphviz inside) to visualize an imported bpmn graph.
    """

    @staticmethod
    def make_graph(data, show_edge_labels=True, vertical=True, orthogonal_lines=False) -> pydotplus.Dot:
        """
        Returns the visualization object of the imported bpmn-graph.

        Parameters
        ----------
        data: DataContainer
            Object that contains a graph read from the file.

        show_edge_labels: boolean, default=False
           If True, edges' labels are displayed in the graph.

        vertical: boolean, default=True
           If True, the direction from the first to the last nodes goes from up to down.

        orthogonal_lines: boolean, default=False
           If true, draws edges that can bend at 90 degrees only, uses splines otherwise.

        Returns
        -------
        pydotplus_graph: pydotplus.Dot
           Graph that can be visualized.
        """
        pydot_node_id_dict = {}  # {id: pydotplus.Node(), ...}

        all_local_nodes = [obj for obj in data.all_objects.values() if isinstance(obj, LocalNode)]
        all_local_edges = [obj for obj in data.all_objects.values() if isinstance(obj, LocalEdge)]
        pydot_graph = pydotplus.Dot(strict=True)
        cluster_dict = {obj.id: pydotplus.Cluster(obj.id, color='lightblue') for obj in all_local_nodes
                        if not obj.is_separate_node}

        # Adding normal (separate) nodes to clusters (subgraphs)
        for node in all_local_nodes:
            if node.is_separate_node:
                node_graph = cluster_dict[node.parent_node.id] if node.parent_node.id in cluster_dict else pydot_graph
                PydotPlusGraphMaker.create_pydot_node(node, node_graph, pydot_node_id_dict)

        # Adding subgraphs to each other (in case one subgraph is inside another)
        proc_list = list(data.all_objects[k] for k in cluster_dict.keys())
        ready_subgraphs = set()
        while len(proc_list) != 0:
            proc = proc_list.pop(0)
            subgraph_list = proc.get_subgraph_objects()
            if not set(subgraph_list).issubset(ready_subgraphs):
                proc_list.append(proc)  # work on it later
            else:  # add it to upper pydot_graph
                upper_proc = proc.parent_node
                upper_graph = pydot_graph if upper_proc is None else cluster_dict[upper_proc.id]
                upper_graph.add_subgraph(cluster_dict[proc.id])
                ready_subgraphs.add(proc)

        # Modify names because graphviz can work incorrectly with specific symbols
        for graph in [pydot_graph] + list(cluster_dict.values()):
            graph.set_name(modify_id(graph.get_name()))
            for n in graph.get_nodes():
                n.set_name(modify_id(n.get_name()))

        # Create pydotplus.Edges
        for edge in all_local_edges:
            if edge.type == 'attachedToRef':
                color = 'pink'
            else:
                color = 'black'
            e = pydotplus.Edge(pydot_node_id_dict[edge.source_node.id],
                               pydot_node_id_dict[edge.target_node.id],
                               color=color)
            if show_edge_labels:
                e.set('label', '\n'.join(edge.get_object_names()))
            pydot_graph.add_edge(e)

        if not vertical:
            pydot_graph.set('rankdir', 'LR')
        if orthogonal_lines:
            pydot_graph.set('splines', 'ortho')

        return pydot_graph

    @staticmethod
    def create_pydot_node(node, graph, pydot_node_id_dict):
        """
        Creates pydot.Node() objects and adds it to parent graph.

        Parameters
        ----------
        node: LocalNode
           Node object.

        graph: pydotplus.Dot
           Pydotplus'es graph object.

        pydot_node_id_dict: dict of {str: pydotplus.Node}
           Dict that contains id of the node and a corresponding pydotplus'es node object.

        """
        if node.type == 'startEvent':
            n = pydotplus.Node(name=node.id, fillcolor="green", style="filled", label=node.name)
        elif node.type == 'endEvent':
            n = pydotplus.Node(name=node.id, fillcolor="red", style="filled", label=node.name)
        elif 'task' in node.type.lower():
            n = pydotplus.Node(name=node.id, shape="box", style="filled", label=node.name)
        elif node.type == 'parallelGateway':
            n = pydotplus.Node(name=node.id, shape="diamond", style="filled", label='+')
        elif node.type == 'exclusiveGateway':
            n = pydotplus.Node(name=node.id, shape="diamond", style="filled", label='x')
        else:
            n = pydotplus.Node(name=node.id, style="filled", label=node.name)
        graph.add_node(n)
        pydot_node_id_dict[node.id] = n


class BpmnGraphMaker:
    """
    Class that transforms DataContainer into BPMN graph.
    """

    @staticmethod
    def make_graph(data) -> Graph:
        """
        Returns the visualization object of the imported bpmn-graph.

        Parameters
        ----------
        data: DataContainer
            Object that contains a graph read from the file.

        Returns
        -------
        graph: Graph
           Graph that can be visualized.
        """

        graph = create_bpmn()

        all_local_nodes = [obj for obj in data.all_objects.values() if isinstance(obj, LocalNode)]
        all_local_edges = [obj for obj in data.all_objects.values() if isinstance(obj, LocalEdge)]

        # Adding normal (separate) nodes to graph
        node_id_set = set()
        for node in all_local_nodes:
            if node.is_separate_node:
                added = True
                node_id = modify_id(node.id)
                if node.type == 'startEvent':
                    graph.add_node(node_id, '', NodeType.START_EVENT)
                elif node.type == 'endEvent':
                    graph.add_node(node_id, '', NodeType.END_EVENT)
                elif 'task' in node.type.lower():
                    graph.add_node(node_id, node.name, NodeType.TASK)
                elif node.type == 'parallelGateway':
                    graph.add_node(node_id, '', NodeType.PARALLEL_GATEWAY)
                elif node.type == 'exclusiveGateway':
                    graph.add_node(node_id, '', NodeType.EXCLUSIVE_GATEWAY)
                else:
                    # some other type that we do not process
                    added = False
                if added:
                    node_id_set.add(node_id)

        # Create pydotplus.Edges
        for edge in all_local_edges:
            id1 = modify_id(edge.source_node.id)
            id2 = modify_id(edge.target_node.id)
            if id1 in node_id_set and id2 in node_id_set:
                graph.add_edge(id1, id2)

        return graph


def modify_id(string):
    """
    Modifies string (id, name,...) because graphviz can work incorrectly with specific symbols.

    Parameters
    ----------
    string: str
       Text string.

    Returns
    -------
    result: str
        Modified string.
    """
    return string.replace('-', '_')
