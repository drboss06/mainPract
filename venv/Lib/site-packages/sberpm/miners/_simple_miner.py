from ._abstract_miner import AbstractMiner
from .._holder import DataHolder
from ..visual._graph import create_dfg, Graph


class SimpleMiner(AbstractMiner):
    """
    Realization of a simple miner algorithm that creates all edges that exist
    according to the event log (no filtration is performed).

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import SimpleMiner
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = SimpleMiner(data_holder)
    >>> miner.apply()
    """

    def __init__(self, data_holder: DataHolder):
        super().__init__(data_holder)

    def apply(self):
        """
        Starts the calculation of the graph using the miner.
        """
        unique_activities = self._data_holder.get_unique_activities()
        follows_pairs = super()._get_follows_pairs()

        graph = create_dfg()
        super().create_act_nodes(graph, unique_activities)
        super().create_start_end_events_and_edges(graph, *super()._get_first_last_activities())
        self.create_edges(graph, follows_pairs)
        self.graph = graph

    @staticmethod
    def create_edges(graph, pairs):
        """
        Adds edges between transitions to the graph.

        Parameters
        ----------
        graph: Graph
            Graph.

        pairs: list of tuple(str, str)
            Pairs of activities that have causal relation.
        """
        for pair in pairs:
            graph.add_edge(pair[0], pair[1])


def simple_miner(data_holder: DataHolder) -> Graph:
    """
    Realization of a simple miner algorithm that creates all edges that exist
    according to the event log (no filtration is performed).

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    Returns
    -------
    graph : Graph

    """
    miner = SimpleMiner(data_holder)
    miner.apply()
    return miner.graph
