import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
from cvxopt import glpk, matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from ._abstract_miner import AbstractMiner
from .._holder import DataHolder
from ..visual._graph import create_dfg, Graph


class CorrelationMiner(AbstractMiner):
    """
    Miner that can create a graph of the event log
    that does not have ID column.

    The graph will have 'count' metric for nodes and edges.


    Parameters
    ----------
    data_holder: DataHolder
        Object that contains the event log and the names of its necessary columns.

    greedy: bool, default=True
        If True, time duration matrix is computed in a greedy way.
        There is probability that the solution will not be optimal,
        but it greatly reduces time and memory for big logs.

    sparse: bool, default=False
        Used only when greedy=False. If True, the computation of
        duration matrix will be done with the use of sparse matrix.
        It can slightly increase computational time, but slightly reduce
        the memory.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import CorrelationMiner
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, None, 'activity_column', 'dt_column')
    >>>
    >>> miner = CorrelationMiner(data_holder)
    >>> miner.apply()

    References
    ----------
    Shaya Pourmirza, Remco Dijkman, and Paul Grefen.
    "Correlation miner: mining business process models and event correlations
    without case identifiers." International Journal of Cooperative Information Systems, 2017
    """

    def __init__(self, data_holder: DataHolder, greedy: bool = True, sparse: bool = False):
        super().__init__(data_holder)
        self._greedy = greedy
        self._sparse = sparse

    def apply(self) -> None:
        """
        Starts the calculation of the graph using the miner.
        """
        act_col = self._data_holder.activity_column
        timestamp_col = self._data_holder.get_timestamp_col()
        data = self._data_holder.data[[act_col, timestamp_col]]
        activities = np.array(sorted(self._data_holder.get_unique_activities()))
        act_counts_dict = data[act_col].value_counts(sort=False).to_dict()
        act_counts_list = [act_counts_dict[a] for a in activities]
        timestamps = {i: data[data[act_col] == act][timestamp_col].astype(int).values / 10 ** 9
                      for i, act in enumerate(activities)}

        ps_matrix = get_precede_succeed_matrix(activities, timestamps)
        duration_matrix = get_duration_matrix(activities, timestamps, self._greedy, self._sparse)
        c_matrix = get_c_matrix(ps_matrix, duration_matrix, act_counts_list)
        solved, edges_count_matrix = solve_lp_problem(c_matrix, act_counts_list)

        if solved:
            # Create graph
            graph = create_dfg()
            super().create_act_nodes(graph, activities)
            graph.add_node_metric('count', act_counts_dict)
            self._create_edges_add_count_metric(graph, activities, edges_count_matrix)
            self.graph = graph

    @staticmethod
    def _create_edges_add_count_metric(graph,
                                       activities: np.ndarray,
                                       edges_count_matrix: np.ndarray):
        """
        Adds edges between transitions to the graph.

        Parameters
        ----------
        graph: Graph
            Graph.

        activities: np.ndndarray of str
            List of activities

        edges_count_matrix: np.ndarray of int, shape=[activities_num, activities_num]
            Adjacency matrix of the graph.

        """
        count_metric = {}
        for i, a1 in enumerate(activities):
            for j, a2 in enumerate(activities):
                if edges_count_matrix[i, j] != 0:
                    graph.add_edge(a1, a2)
                    count_metric[(a1, a2)] = edges_count_matrix[i, j]
        graph.add_edge_metric('count', count_metric)


def get_precede_succeed_matrix(activities: np.ndarray,
                               timestamps: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Calculate PS matrix.

    Parameters
    ----------
    activities: np.ndarray of str
        Names of unique activities.

    timestamps: Dict[int, np.ndarray]
        Dict of timestamps for each activity. Int number corresponds
        to the index of activity in 'activities' array.

    Returns
    -------
    ps_matrix: np.ndarray of float, shape=[len(activities), len(activities)]

    """
    ps_matrix = np.zeros((len(activities), len(activities)), dtype='float')

    # This code is valid if there are no duplicate timestamps.
    # As long as it is not guarantied, better to calc each element separately.
    # So, if there are duplicate timestamps,
    # ps_matrix[i, j] + ps_matrix[j, i] might not be equal 1.

    # calculate upper triangular part of matrix
    # for i in range(len(activities)):
    #     ts1 = timestamps[i]
    #     for j in range(i + 1, len(activities)):
    #         ts2 = timestamps[j]
    #         value = calc_ps_matrix_value(ts1, ts2)
    #         ps_matrix[i, j] = value

    # # fill lower triangular part of matrix
    # for i in range(1, len(activities)):
    #     for j in range(i):
    #         ps_matrix[i, j] = 1 - ps_matrix[j, i]

    for i in range(len(activities)):
        ts1 = timestamps[i]
        for j in range(len(activities)):
            if i != j:
                ts2 = timestamps[j]
                value = calc_ps_matrix_value(ts1, ts2)
                ps_matrix[i, j] = value

    return ps_matrix


def calc_ps_matrix_value(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate one value of ps+matrix.

    Parameters
    ----------
    ts1: np.ndarray
        Timestamps of activity_1.

    ts2: np.ndarray
        Timestamps of activity_2.

    Returns
    -------
    value: float

    """
    total_combinations = len(ts1) * len(ts2)
    p, q = 0, 0
    follows = 0
    while p < len(ts1):
        while q < len(ts2):
            if ts1[p] < ts2[q]:
                break
            q += 1
        follows += len(ts2) - q
        p += 1
    value = follows / total_combinations
    return value


def get_duration_matrix(activities: np.ndarray,
                        timestamps: Dict[int, np.ndarray],
                        greedy: bool,
                        sparse) -> np.ndarray:
    """
    Calculate time duration matrix.

    Parameters
    ----------
    activities: np.ndarray of str
        Names of unique activities.

    timestamps: Dict[int, np.ndarray of int]
        Dict of timestamps for each activity. Int number corresponds
        to the index of activity in 'activities' array.

    greedy: bool
    sparse: bool

    Returns
    -------
    duration_matrix: np.ndarray of float, shape=[len(activities), len(activities)]

    """
    duration_matrix = np.zeros((len(activities), len(activities)), dtype='float')

    for i in range(len(activities)):
        ts1 = timestamps[i]
        for j in range(len(activities)):
            if i != j:
                ts2 = timestamps[j]
                duration_matrix[i, j] = calc_mean_duration_value(ts1, ts2, greedy, sparse)
    return duration_matrix


def calc_mean_duration_value(ts1: np.ndarray,
                             ts2: np.ndarray,
                             greedy: bool,
                             sparse: bool) -> float:
    """
    Calculate one value of a duration matrix.

    Parameters
    ----------
    ts1: np.ndarray
        Timestamps of activity_1.

    ts2: np.ndarray
        Timestamps of activity_2.

    greedy: bool
    sparse: bool

    Returns
    -------
    value: float

    """
    ts1, ts2 = clean_ts(ts1, ts2)

    if greedy:
        # Two greedy algorithms are used. The minimum value of the two is returned.
        # Example:
        #   ts1: 1   3   5.
        #   ts2:   2   4   6   7.
        #   Greedy left pairs:  (1, 2), (3, 4), (5, 6); mean = (1+1+1)/3 = 1 (returned)
        #   Greedy right pairs: (5, 7), (3, 6), (1, 4); mean = (2+3+3)/3 = 2.(6)

        # Greedy left
        # For each ts1[i] the closest free ts2[j] is found (ts2[j] > ts1[i])
        def greedy_left(ts1, ts2) -> float:
            left_durations = []
            i = 0
            j = 0
            while i < len(ts1):
                while j < len(ts2):
                    if ts1[i] < ts2[j]:
                        left_durations.append(ts2[j] - ts1[i])
                        j = j + 1
                        break
                    j = j + 1
                i = i + 1
            return np.mean(left_durations)

        # Greedy right
        # For each ts2[j] the closest free ts1[i] is found (ts2[j] > ts1[i])
        def greedy_right(ts1, ts2) -> float:
            right_durations = []
            i = len(ts1) - 1
            j = len(ts2) - 1
            while j >= 0:
                while i >= 0:
                    if ts1[i] < ts2[j]:
                        right_durations.append(ts2[j] - ts1[i])
                        i = i - 1
                        break
                    i = i - 1
                j = j - 1
            return np.mean(right_durations)

        greedy_left_mean = greedy_left(ts1, ts2)
        greedy_right_mean = greedy_right(ts1, ts2)

        return min([greedy_left_mean, greedy_right_mean])

    else:
        if sparse:
            biadjacency_matrix = create_sparse_matrix(ts1, ts2)
            row_indexes, col_indexes = min_weight_full_bipartite_matching(biadjacency_matrix)
        else:
            biadjacency_matrix = create_dense_matrix(ts1, ts2)
            row_indexes, col_indexes = linear_sum_assignment(biadjacency_matrix)

        return biadjacency_matrix[row_indexes, col_indexes].mean()


def create_sparse_matrix(ts1: np.ndarray, ts2: np.ndarray) -> csr_matrix:
    """
    Create sparse matrix of time durations.
    Negative duration values will converted to zeros.

    Parameters
    ----------
    ts1: np.ndarray
        Timestamps of activity_1.

    ts2: np.ndarray
        Timestamps of activity_2.

    Returns
    -------
    matrix: csr_matrix, shape=[len(ts1), len(ts2)]
    """
    diff = ts2[np.newaxis, :] - ts1[:, np.newaxis]
    diff[diff < 0] = 0
    return csr_matrix(diff)


def create_dense_matrix(ts1: np.ndarray, ts2: np.ndarray) -> np.ndarray:
    """
    Create np.ndarray matrix of time durations.
    Negative duration values will converted np.inf.

    Parameters
    ----------
    ts1: np.ndarray
        Timestamps of activity_1.

    ts2: np.ndarray
        Timestamps of activity_2.

    Returns
    -------
    matrix: np.ndarray, shape=[len(ts1), len(ts2)]
    """
    diff = ts2.astype(float)[np.newaxis, :] - ts1[:, np.newaxis]
    diff[diff < 0] = np.inf
    return diff


def get_c_matrix(ps_matrix: np.ndarray,
                 duration_matrix: np.ndarray,
                 act_counts_list: List[int]) -> np.ndarray:
    """
    Calculate matrix with coefficients of the linear objective function.

    Parameters
    ----------
    ps_matrix: np.ndarray
    duration_matrix: np.ndarray
    act_counts_list: list of int

    Returns
    -------
    c_matrix: np.ndarray, shape=len(activities), len(activities)
    """
    c_matrix = np.zeros((len(act_counts_list), len(act_counts_list)))
    for i in range(len(act_counts_list)):
        for j in range(len(act_counts_list)):
            value = duration_matrix[i, j] / ps_matrix[i, j] * \
                    1 / (min(act_counts_list[i], act_counts_list[j])) if ps_matrix[i, j] != 0 else 0
            if value == 0:
                value = 10 ** 11  # this value has a big influence on the resulting graph!!!
            c_matrix[i, j] = value

    return c_matrix


def solve_lp_problem(c_matrix: np.ndarray,
                     act_counts_list: List[int]) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Solve Integer Linear Programming problem.

    Parameters
    ----------
    c_matrix: np.ndarray
        Coefficients of the linear objective function.

    act_counts_list: list of int

    Returns
    -------
    solved: bool
        True, if solved successfully, False otherwise.

    edges_count_matrix: np.ndarray of int, shape=[len(activities), len(activities)] or None
        If solved==True, result of the problem: counts of edges.
        If solved==False, None.
    """
    act_num = len(act_counts_list)

    # 1. Equality constraints

    # Every line of A_eq_outgoing says that sum of the outgoing edges from
    # the given activity must be equal to this activity count in the event log.
    # Columns represent possible edges.
    # Example:
    # |aa|ab|ba|bb| count |
    # | 1| 1| 0| 0|a_count|  num_edges(a, a) + num_edges(a, b) = a_count
    # | 0| 0| 1| 1|b_count|  num_edges(b, a) + num_edges(b, b) = b_count
    separate_matrixes = []
    for i in range(act_num):
        m = np.zeros((act_num, act_num))
        m[i, :] = [1] * act_num
        separate_matrixes.append(m)
    A_eq_outgoing = np.concatenate(separate_matrixes, axis=1)

    # Every line of A_eq_incoming says that sum of the incoming edges to
    # the given activity must be equal to this activity count in the event log.
    # Columns represent possible edges.
    # Example:
    # |aa|ab|ba|bb| count |
    # | 1| 0| 1| 0|a_count|  num_edges(a, a) + num_edges(b, a) = a_count
    # | 0| 1| 0| 1|b_count|  num_edges(a, b) + num_edges(b, b) = b_count
    A_eq_incoming = np.concatenate([np.diag([1] * act_num) for _ in range(act_num)], axis=1)

    A_eq = np.concatenate([A_eq_outgoing, A_eq_incoming], axis=0)
    b_eq = np.concatenate([act_counts_list, act_counts_list])
    b_eq = b_eq.astype('float')

    # 2. bounds
    # Specify that:
    # a) an edge(a, b) count value must be <=min(a_count, b_count)
    # b) an edge(a, b) count value must be >=0 and
    # bounds = [(0, min(act_counts_list[i], act_counts_list[j]))
    #           for i in range(act_num) for j in range(act_num)]

    # a)
    A_ub_min = np.diag([1.] * act_num ** 2)  # each row - one unique edge
    b_ub_min = np.zeros(act_num ** 2, dtype='float')
    for i in range(act_num):
        for j in range(act_num):
            edge_ind = i * act_num + j
            b_ub_min[edge_ind] = min([act_counts_list[i], act_counts_list[j]])
    # b)
    Aub_zero = -A_ub_min
    b_ub_zero = np.zeros(act_num ** 2, dtype='float')

    A_ub = np.concatenate([A_ub_min, Aub_zero], axis=0)
    b_ub = np.concatenate([b_ub_min, b_ub_zero])

    c_matrix = c_matrix.flatten().tolist()

    # convert all the input data
    c_matrix = matrix(c_matrix)
    A_ub = matrix(A_ub)
    b_ub = matrix(b_ub)
    A_eq = matrix(A_eq)
    b_eq = matrix(b_eq)

    status, x, _, _ = glpk.lp(c_matrix, A_ub, b_ub, A_eq, b_eq)
    if status == 'optimal':
        edges_count_matrix = np.array(x).reshape(act_num, act_num).astype('int')
        return True, edges_count_matrix
    else:
        warnings.simplefilter('always', RuntimeWarning)
        warnings.warn(f"Linear program solving failed (finished with status '{status}')."
                      f"The graph cannot be built.", RuntimeWarning)
        return False, None


def clean_ts(ts1: np.ndarray, ts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove some timestamps that definitely will not be matched.
    It will guarantee the absence of errors during linear sum assignment
    problem solving by scipy and better result by greedy algorithm.

    Parameters
    ----------
    ts1: np.ndarray
    ts2: np.ndarray

    Returns
    -------
    ts1, ts2
    """
    # 1. Remove timestamps:
    # - of ts2 that are smaller than min(ts1),
    # - of ts1 that are bigger than max(ts2)
    if ts2[0] < ts1[0]:
        ts2 = ts2[ts2 > ts1[0]]
    if ts1[-1] > ts2[-1]:
        ts1 = ts1[ts1 < ts2[-1]]

    # 2. Remove middle elements of ts2
    new_ts2 = []
    i, j = 0, 0
    while i < len(ts1) and j < len(ts2):
        if ts1[i] <= ts2[j]:
            new_ts2.append(ts2[j])
            i += 1
            j += 1
        else:
            # move right on ts2 chain until ts2 element is bigger that ts1 element
            j += 1
    new_ts2 = np.concatenate([new_ts2, ts2[j:]])  # add the ts2[j:] too (can be empty if j is maximum)

    # 3. Remove middle elements of ts1
    new_ts1 = []
    i, j = len(ts1) - 1, len(new_ts2) - 1
    while j >= 0 and i >= 0:
        if ts1[i] <= new_ts2[j]:
            new_ts1.append(ts1[i])
            i -= 1
            j -= 1
        else:
            # move left on ts1 chain until ts1 element is smaller that ts2 element
            i -= 1
    new_ts1 = np.concatenate([ts1[:i + 1], new_ts1[::-1]])  # add the ts1[:i + 1] too (can be empty if i==-1)

    return new_ts1, new_ts2


def correlation_miner(data_holder: DataHolder, greedy: bool = True, sparse: bool = False) -> Graph:
    """
    Miner that can create a graph of the event log
    that does not have ID column.

    The graph will have 'count' metric for nodes and edges.


    Parameters
    ----------
    data_holder: DataHolder
        Object that contains the event log and the names of its necessary columns.

    greedy: bool, default=True
        If True, time duration matrix is computed in a greedy way.
        There is probability that the solution will not be optimal,
        but it greatly reduces time and memory for big logs.

    sparse: bool, default=False
        Used only when greedy=False. If True, the computation of
        duration matrix will be done with the use of sparse matrix.
        It can slightly increase computational time, but slightly reduce
        the memory.

    Returns
    -------
    graph : Graph

    References
    ----------
    Shaya Pourmirza, Remco Dijkman, and Paul Grefen.
    "Correlation miner: mining business process models and event correlations
    without case identifiers." International Journal of Cooperative Information Systems, 2017

    """
    miner = CorrelationMiner(data_holder, greedy, sparse)
    miner.apply()
    return miner.graph
