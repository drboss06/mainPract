from typing import Tuple

import numpy as np
import pandas as pd

from ._abstract_miner import AbstractMiner
from .._holder import DataHolder
from ..visual._graph import create_dfg, Graph


class HeuMiner(AbstractMiner):
    """
    Realization of Heuristic Miner algorithm.
    This algorithm is used to filter only the most "important" edges between the nodes.

    Parameters
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    threshold : float, default=0.8
        Parameter of Heuristic miner. Ranges from 0 to 1.
        The bigger the threshold, the less edges will remain in the graph.

        For every edge the "importance" coefficient will be calculated, it ranges from -1 to 1.
        If it will be equal to or higher than the threshold, the edge will remain,
        otherwise it will be removed.

    Attributes
    ----------
    threshold: float
        Parameter of Heuristic miner. Ranges from 0 to 1.

    df_pairs: pd.DataFrame or None
        Each row represents a unique pair of two activities (edge),
        contains the resulting coefficient and temporary data.
        Columns:
        'a', 'b' - first and second activities in a pair,
        'a>b', 'b>a' - number of 'ab' and 'ba' pairs in the log,
        'coeff' - resulting coefficient.

    df_triples: pd.DataFrame or None
        Each row represents a unique triple of activities
        in a length-two loop, like [a, b, a...].
        It contains the resulting coefficient and temporary data.
        Columns:
        'a', 'b' - first and second activities in a triple (no need to store the third one),
        'a>>b', 'b>>a' - number of 'aba' and 'bab' triples in the log,
        'coeff' - resulting coefficient.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import HeuMiner
    >>>
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = HeuMiner(data_holder)
    >>> miner.apply()

    Notes
    -----
    This implementation includes the basic idea of calculating coefficients for edges and
    selecting the "important" ones using a threshold. It can also deal with cycles of lengths one and two.

    Some other possible features of the miner: the ability of heuristic miner to detect parallel activities,
    mining long-distant dependencies, noise cleaning -  are not implemented here.

    References
    ----------
    A.J.M.M. Weijters, W.M.P van der Aalst, and A.K. Alves de Medeiros.
    Process Mining with the Heuristics Miner-Algorithm, 2006

    https://pdfs.semanticscholar.org/1cc3/d62e27365b8d7ed6ce93b41c193d0559d086.pdf
    """

    def __init__(self, data_holder: DataHolder, threshold: float = 0.8):
        super().__init__(data_holder)
        self.threshold = threshold
        self.df_pairs, self.df_triples = None, None

    def apply(self) -> None:
        """
        Starts the calculation of the graph using the heuristic miner.
        """
        unique_activities = self._data_holder.get_unique_activities()

        self.df_pairs, self.df_triples = self._calc_coeffs()

        graph = create_dfg()
        super().create_act_nodes(graph, unique_activities)
        super().create_start_end_events_and_edges(graph, *super()._get_first_last_activities())

        self._create_edges(graph)
        self.graph = graph

    def _calc_coeffs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates coefficients for edges.

        Returns
        -------
        df_pairs: pd.DataFrame
            It contains coeffs for all edges, including length-one loops.
            Columns: [a, b, a>b, b>a, coeff]

        df_triples: pd.DataFrame
            It contains coeffs for triples (for length-two loops).
            Columns: [a, b, a>>b, b>>a, coeff]
        """
        df = pd.DataFrame()
        df['id'] = self._data_holder.data[self._data_holder.id_column]
        df['a'] = self._data_holder.data[self._data_holder.activity_column]
        df['b'] = df.groupby('id')['a'].shift(-1)

        # normal and length-one loops
        df_pairs = df.groupby(['a', 'b']).count() \
            .reset_index().rename(columns={'id': 'a>b'})  # [a, b, a>b]
        df_pairs = get_reversed_values(df_pairs, 'a>b', 'b>a')  # [a, b, a>b, b>a]

        df_pairs['coeff'] = np.where(
            df_pairs['a'] != df_pairs['b'],
            (df_pairs['a>b'] - df_pairs['b>a']) / (df_pairs['a>b'] + df_pairs['b>a'] + 1),  # normal
            df_pairs['a>b'] / (df_pairs['a>b'] + 1))  # length-one loops
        # [a, b, a>b, b>a, coeff]

        # length-two loops
        df_triples = df
        df_triples['c'] = df_triples.groupby('id')['a'].shift(-3)

        df_triples = df_triples[
            (df_triples['a'] == df_triples['c']) & (df_triples['a'] != df_triples['b'])
            ]
        df_triples = df_triples.groupby(['a', 'b'])['id'].count() \
            .reset_index().rename(columns={'id': 'a>>b'})  # [a, b, a>>b]
        df_triples = get_reversed_values(df_triples, 'a>>b', 'b>>a')  # [a, b, a>>b, b>>a]
        df_triples['coeff'] = \
            (df_triples['a>>b'] + df_triples['b>>a']) / (df_triples['a>>b'] + df_triples['b>>a'] + 1)
        # [a, b, a>>b, b>>a, coeff]

        return df_pairs, df_triples

    def _create_edges(self, graph) -> None:
        """
        Adds nodes and edges to the graph.
        """
        df_pairs = self.df_pairs[self.df_pairs['coeff'] >= self.threshold][['a', 'b']]
        pairs = set(zip(df_pairs['a'], df_pairs['b']))
        for a, b in pairs:
            graph.add_edge(a, b)

        # length-two loops
        df_triples = self.df_triples[self.df_triples['coeff'] >= self.threshold][['a', 'b']]
        triples = set(zip(df_triples['a'], df_triples['b']))
        for a, b in triples:
            if (a, b) not in graph.edges:
                graph.add_edge(a, b)
            if (b, a) not in graph.edges:
                graph.add_edge(b, a)


def get_reversed_values(df: pd.DataFrame, v1_col: str, v2_col: str) -> pd.DataFrame:
    """
    Gets a dataframe with edges and their counts.
    It must have 3 columns: 'a' (source node), 'b' (target node)
    and a third column that represents edges' counts.
    Returns the same dataframe with an additional column
    that contains counts of reversed edges.

    Parameters
    ----------
    df: pd.DataFrame
        Columns: ['a', 'b', v1_col].

    v1_col: str
        Name of the column with edges' counts (must be present in df).

    v2_col: str
        Name of the column with reversed edges' counts
        (must NOT be present in df, will be created in this method).

    Returns
    -------
    df: pd.DataFrame
        Columns: ['a', 'b', v1_col, v2_col].

    """
    df['ab'] = list(zip(df['a'], df['b']))
    df['ba_temp'] = list(zip(df['b'], df['a']))

    temp_df = df[['ab', v1_col]].set_index('ab').rename(columns={v1_col: v2_col})
    df = df.set_index('ba_temp').join(temp_df, how='left').reset_index(drop=True)
    df[v2_col] = df[v2_col].fillna(0)

    return df[['a', 'b', v1_col, v2_col]]


def heu_miner(data_holder: DataHolder, threshold: float = 0.8) -> Graph:
    """
    Realization of Heuristic Miner algorithm.
    This algorithm is used to filter only the most "important" edges between the nodes.

    Parameters
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    threshold : float, default=0.8
        Parameter of Heuristic miner. Ranges from 0 to 1.
        The bigger the threshold, the less edges will remain in the graph.

        For every edge the "importance" coefficient will be calculated, it ranges from -1 to 1.
        If it will be equal to or higher than the threshold, the edge will remain,
        otherwise it will be removed.

    Returns
    -------
    graph : Graph

    Notes
    -----
    This implementation includes the basic idea of calculating coefficients for edges and
    selecting the "important" ones using a threshold. It can also deal with cycles of lengths one and two.

    Some other possible features of the miner: the ability of heuristic miner to detect parallel activities,
    mining long-distant dependencies, noise cleaning -  are not implemented here.

    References
    ----------
    A.J.M.M. Weijters, W.M.P van der Aalst, and A.K. Alves de Medeiros.
    Process Mining with the Heuristics Miner-Algorithm, 2006

    https://pdfs.semanticscholar.org/1cc3/d62e27365b8d7ed6ce93b41c193d0559d086.pdf
    """
    miner = HeuMiner(data_holder, threshold)
    miner.apply()
    return miner.graph
