class GraphType:
    """
    Types of graph
    """
    PETRI_NET = 'Petri-Net'
    DFG = 'DFG'
    BPMN = 'BPMN'


class NodeType:
    """
    Types of nodes
    """
    START_EVENT = 'startevent'
    END_EVENT = 'endevent'
    TASK = 'task'
    PLACE = 'place'
    PARALLEL_GATEWAY = 'parallel_gateway'
    EXCLUSIVE_GATEWAY = 'exclusive_gateway'
