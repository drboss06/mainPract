from ._abstract_miner import AbstractMiner
from .._holder import DataHolder
from ..visual._graph import create_dfg, Graph


class CausalMiner(AbstractMiner):
    """
    Realization of a simple miner algorithm that creates edges only between activities that have causal relation.

    Parameters
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import CausalMiner
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = CausalMiner(data_holder)
    >>> miner.apply()
    
    Notes
    -----
    "activity_1" and "activity_2" have causal relation ("activity_1 -> activity_2") if
        "activity_1 > activity_2" and not "activity_2 > activity_1"
    "activity_1 > activity_2" means that "activity_1" is directly followed by "activity_2" in at least
        one event trace in the event log.
    """

    def __init__(self, data_holder):
        super().__init__(data_holder)

    def apply(self):
        """
        Starts the calculation of the graph using the miner.
        """
        unique_activities = self._data_holder.get_unique_activities()
        follows_pairs = super()._get_follows_pairs()
        causal_pairs, _ = super()._get_causal_parallel_pairs(follows_pairs)

        graph = create_dfg()
        super().create_act_nodes(graph, unique_activities)
        super().create_start_end_events_and_edges(graph, *super()._get_first_last_activities())
        self.create_edges(graph, causal_pairs)
        self.graph = graph

    @staticmethod
    def create_edges(graph, causal_pairs):
        """
        Adds edges between transitions to the graph.

        Parameters
        ----------
        graph: Graph
            Graph.

        causal_pairs: list of tuple(str, str)
            Pairs of activities that have causal relation.
        """
        for pair in causal_pairs:
            graph.add_edge(pair[0], pair[1])


def causal_miner(data_holder: DataHolder) -> Graph:
    """
    Realization of a simple miner algorithm that creates edges
    only between activities that have causal relation.

    Parameters
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    Returns
    -------
    graph : Graph

    Notes
    -----
    "activity_1" and "activity_2" have causal relation ("activity_1 -> activity_2") if
        "activity_1 > activity_2" and not "activity_2 > activity_1"
    "activity_1 > activity_2" means that "activity_1" is directly followed by "activity_2" in at least
        one event trace in the event log.
    """
    miner = CausalMiner(data_holder)
    miner.apply()
    return miner.graph
