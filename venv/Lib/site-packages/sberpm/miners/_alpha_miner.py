import multiprocessing as mp
from copy import deepcopy

import numpy as np

from ._abstract_miner import AbstractMiner
from .._holder import DataHolder
from ..visual._graph import create_petri_net, Graph
from ..visual._types import NodeType


class AlphaMiner(AbstractMiner):
    """
    Realization of an Alpha Miner algorithm.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    n_jobs: int, default=1
        Maximum number of processes created if possible.
        If n_jobs > 0 - max number of processes = n_jobs;
        if n_jobs < 0 - max number of processes = mp.cpu_count() + 1 + n_jobs;
        in other cases: max number of processes = 1.

    Attributes
    ----------
    _n_jobs: : int, default=1
        Maximum number of processes created if possible.
        If n_jobs > 0 - max number of processes = n_jobs;
        if n_jobs < 0 - max number of processes = mp.cpu_count() + 1 + n_jobs;
        in other cases: max number of processes = 1.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import AlphaMiner
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = AlphaMiner(data_holder)
    >>> miner.apply()

    Notes
    -----
    This algorithm can not handle loops of lengths 1.

    References
    ----------
    W.M.P. van der Aalst, A.J.M.M. Weijters, and L. Maruster. Workflow Mining: Discovering Process Models
    from Event Logs. IEEE Transactions on Knowledge and Data Engineering (TKDE), Accepted for publication, 2003
    """

    def __init__(self, data_holder, n_jobs=1):
        super().__init__(data_holder)
        self._n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count() + 1 + n_jobs if n_jobs < 0 else 1

    def apply(self):
        """
        Starts the calculation of the graph using the miner.
        """
        unique_activities = self._data_holder.get_unique_activities()
        follows_pairs = super()._get_follows_pairs()

        places = self.find_places(follows_pairs, unique_activities)

        # Create and fill the graph
        graph = create_petri_net()
        super().create_act_nodes(graph, unique_activities)
        super().create_start_end_events_and_edges(graph, *super()._get_first_last_activities())
        self.create_places_and_edges(graph, places)

        self.graph = graph

    def find_places(self, follows_pairs, unique_activities):
        """
        Finds places (pairs of sets of activities).

        Parameters
        ----------
        follows_pairs: set of tuple(str, str)
            Pairs of activities that follow each other in the event log.

        unique_activities: array-like of str
            List of activities.

        Returns
        -------
        places: list of tuple of two sets of str
            List of places. A place is represented by two sets of activities (incoming and outgoing).
        """
        causal_pairs, parallel_pairs = super()._get_causal_parallel_pairs(follows_pairs)
        label2act, act2label = super()._label_encode_activities(unique_activities)
        independence_matrix = self._create_independence_matrix(causal_pairs, parallel_pairs, act2label)
        places = self._calculate(causal_pairs, independence_matrix, act2label, label2act, self._n_jobs)
        return places

    @staticmethod
    def _create_independence_matrix(causal_pairs, parallel_pairs, act2label):
        """
        Creates a matrix that stores the information whether two activities are independent ('#')
        (do not have causal ('->') or parallel ('||') relations).

        Parameters
        ----------
        causal_pairs : set of tuples (str, str)
            Pairs (activity_1, activity_2) that have causal relation: "activity_1 -> activity_2"

        parallel_pairs : set of tuples (str, str)
            Pairs (activity_1, activity_2) that have parallel relation: "activity_1 || activity_2"

        act2label: dict of {str: int}
            Key: name of an activity. Value: numeric label of the activity.

        Returns
        -------
        independence_matrix: ndarray of bool, shape=[len_of_activities, len_of_activities]
            If activities i and j are independent ('#'), matrix[i][j] is True, False otherwise ('->' or '||').
        """
        independence_matrix = np.full((len(act2label), len(act2label)), True, dtype=bool)

        for pair in causal_pairs.union(parallel_pairs):
            i = act2label[pair[0]]
            j = act2label[pair[1]]
            independence_matrix[i][j] = False
            independence_matrix[j][i] = False

        return independence_matrix

    @staticmethod
    def _calculate(causal_pairs, independence_matrix, act2label, label2act, n_jobs):
        """
        Applies Alpha Miner algorithm: finds pairs of biggest unrelated sets of activities.

        Parameters
        ----------
        causal_pairs : set of tuples (str, str)
            Pairs (activity_1, activity_2) that have causal relation: "activity_1 -> activity_2".

        independence_matrix: ndarray of bool, shape=[len_of_activities, len_of_activities]
            If activities i and j are independent ('#'), matrix[i][j] is True, False otherwise ('->' or '||').

        act2label : dict of {int: str}
            Key: numeric label of an activity. Value: name of the activity.

        label2act : dict of {str: int}
            Key: name of an activity. Value: numeric label of the activity.

        n_jobs: : int, default=1
            Maximum number of processes created if possible.

        Returns
        -------
        places: list of tuple of two sets of str
            List of places. A place is represented by two sets of activities (incoming and outgoing).
        """
        causal_pairs_encoded = []  # will be an ndarray of int, shape=[len of causal_pairs, 2]
        gen0 = []  # list of tuples, containing 2 frozensets
        for pair in causal_pairs:
            enc_act_1 = act2label[pair[0]]
            enc_act_2 = act2label[pair[1]]
            # Keep only those causal pairs, where each activity is not parallel to itself.
            if independence_matrix[enc_act_1][enc_act_1] and independence_matrix[enc_act_2][enc_act_2]:
                causal_pairs_encoded.append([enc_act_1, enc_act_2])
                gen0.append((frozenset([enc_act_1]), frozenset([enc_act_2])))
        causal_pairs_encoded = np.array(causal_pairs_encoded)

        generations = [gen0]  # list of lists of pairs
        while True:
            prev_gen = generations[-1]
            new_pairs, indexes_of_parents_to_remove = AlphaMiner._calc_new_pairs(causal_pairs_encoded, prev_gen,
                                                                                 independence_matrix, n_jobs)
            if len(new_pairs) != 0:
                # Remove the pairs that participated in the creation of the pairs with bigger sets of activities.
                for ind in sorted(indexes_of_parents_to_remove, reverse=True):
                    del prev_gen[ind]
                generations.append(list(new_pairs))
            else:
                break

        all_pairs = []
        for gen in generations:
            all_pairs += [({label2act[act] for act in pair[0]}, {label2act[act] for act in pair[1]}) for pair in gen]
        return all_pairs

    @staticmethod
    def _calc_new_pairs(causal_pairs_encoded, list_of_pairs, independence_matrix, n_jobs, iterations_per_batch=10000):
        """
        Decides whether to run a single-process or a multi-process execution (if possible) to create new pairs.

        Parameters
        ----------
        causal_pairs_encoded: ndarray of int, shape=[causal_pairs_len, 2]
            Causal pairs, each line [a, b] represents a pair a->b (elements - number-encoded activities).

        list_of_pairs: list of tuple of 2 frozensets of int
            List of pairs. Each pair consists of 2 frozensets.
            Each frozenset contains number-encoded activities.

        independence_matrix: ndarray of bool, shape=[unique_act_num, unique_act_num]
            If for activities i and j independence_matrix[i][j] == True, i and j are independent ('#'),
            if False - dependent ('->', '<-', or '||').

        n_jobs: int
            Maximum number of processes created if possible.

        iterations_per_batch: int, default=10000
            Number of iterations = len(causal_pairs_encoded) * len(list_of_pairs).
            If this number is bigger than iterations_per_batch, the algorithm will run in parallel if possible,
            and in that case each process will deal with no more than iterations_per_batch iterations.

        Returns
        -------
        new_pairs: set of tuples of 2 frozensets of int
            New pairs. Each pair consists of 2 frozensets of number-encoded activities.

        indexes_of_parents_to_remove: list of int
            Returns the indexes of "parent" pairs that need to be removed because they participated
            in the creation of "child" pairs with bigger sets of activities.
            If list_of_pairs is a sublist of a bigger one, returns the indexes of
            the elements in a bigger list.
        """
        if len(causal_pairs_encoded) > iterations_per_batch:  # very unlikely
            iterations_per_batch = len(causal_pairs_encoded)
        total_iterations = len(list_of_pairs) * len(causal_pairs_encoded)
        if total_iterations < iterations_per_batch or n_jobs == 1:
            new_pairs, indexes_of_parents_to_remove = \
                AlphaMiner.make_new_pairs(causal_pairs_encoded, list_of_pairs, 0, independence_matrix)
        else:
            num_batches = int(total_iterations / iterations_per_batch) + 1
            columns_per_batch = int(len(list_of_pairs) / num_batches) + 1
            pool = mp.Pool(min(num_batches, n_jobs))
            result_objects = [pool.apply_async(AlphaMiner.make_new_pairs, args=arg) for arg in
                              AlphaMiner._generate_args(causal_pairs_encoded, list_of_pairs, columns_per_batch,
                                                        independence_matrix)]
            new_pairs = set()
            indexes_of_parents_to_remove = []
            for res in result_objects:
                n_pairs, n_indexes = res.get()
                new_pairs = new_pairs.union(n_pairs)
                indexes_of_parents_to_remove += n_indexes

            pool.close()
            pool.join()

        return new_pairs, indexes_of_parents_to_remove

    @staticmethod
    def _generate_args(causal_pairs_encoded, list_of_pairs, columns_per_batch, independence_matrix):
        """
        Generates arguments for make_new_pairs() function in case of parallel execution.
        The main point is to split given data into batches.

        Parameters
        ----------
        causal_pairs_encoded: ndarray of int, shape=[causal_pairs_len, 2]
            Causal pairs, each line [a, b] represents a pair a->b (elements - number-encoded activities).

        list_of_pairs: list of tuple of 2 frozensets of int
            List of pairs. Each pair consists of 2 frozensets.
            Each frozenset contains number-encoded activities.

        columns_per_batch: int
            Maximum umber of columns of list_of_pairs list that wll be present in one batch.

        independence_matrix: ndarray of bool, shape=[unique_act_num, unique_act_num]
            If for activities i and j independence_matrix[i][j] == True, i and j are independent ('#'),
            if False - dependent ('->', '<-', or '||').

        Returns
        -------
        causal_pairs_encoded: ndarray of int, shape=[causal_pairs_len, 2]
            Causal pairs, each line [a, b] represents a pair a->b (elements - number-encoded activities).

        sublist_of_pairs: ndarray of frozensets of int, shape=[num_of_pairs, 2]
            Part of a list_of_pairs list.

        starting_index: int
            Index of a list_of_pairs list where sublist_of_pairs starts.

        independence_matrix: ndarray of bool, shape=[unique_act_num, unique_act_num]
            If for activities i and j independence_matrix[i][j] == True, i and j are independent ('#'),
            if False - dependent ('->', '<-', or '||').
        """
        n = 0
        while n < len(list_of_pairs):
            n += columns_per_batch
            yield (causal_pairs_encoded,
                   np.array(list_of_pairs[n - columns_per_batch: n]),
                   n - columns_per_batch,
                   independence_matrix
                   )

    @staticmethod
    def make_new_pairs(causal_pairs_encoded, list_of_pairs, starting_index, independence_matrix):
        """
        Creates new pairs by trying to combine the elements of two given sets of pairs.

        Parameters
        ----------
        causal_pairs_encoded: ndarray of int, shape=[causal_pairs_len, 2]
            Causal pairs, each line [a, b] represents a pair a->b (elements - number-encoded activities).

        list_of_pairs: ndarray of frozensets of int, shape=[num_of_pairs, 2]
            List of pairs. Each pair consists of 2 frozensets.
            Each frozenset contains number-encoded activities.

        starting_index:
            List_of_pairs might be a sublist of a bigger one, in that case
            starting_index is the index in a bigger list
            of the first element of a given sublist (bigger_list[starting_index] == list_of_pairs[0]).
            If a full list_of_pairs is given, the starting_index must be 0.

        independence_matrix: ndarray of bool, shape=[unique_act_num, unique_act_num]
            If for activities i and j independence_matrix[i][j] == True, i and j are independent ('#'),
            if False - dependent ('->', '<-', or '||').

        Returns
        -------
        new_pairs: set of tuples of 2 frozensets of int
            New pairs. Each pair consists of 2 frozensets of number-encoded activities.

        indexes_of_parents_to_remove: list of int
            Returns the indexes of "parent" pairs that need to be removed because they participated
            in the creation of "child" pairs with bigger sets of activities.
            If list_of_pairs is a sublist of a bigger one, returns the indexes of
            the elements in a bigger list.
        """
        new_pairs = set()
        indexes_of_parents_to_remove = []
        for j, pair2 in enumerate(list_of_pairs):
            not_put_for_removal = True
            for pair1 in causal_pairs_encoded:
                in1 = pair1[0] in pair2[0]
                in2 = pair1[1] in pair2[1]
                if in1 and in2 or not in1 and not in2:
                    continue
                if in1 and not in2 and AlphaMiner._independent(pair1[1], pair2[1], independence_matrix):
                    half1 = deepcopy(pair2[0])
                    half2 = frozenset([el for el in pair2[1]] + [pair1[1]])
                elif not in1 and in2 and AlphaMiner._independent(pair1[0], pair2[0], independence_matrix):
                    half1 = frozenset([el for el in pair2[0]] + [pair1[0]])
                    half2 = deepcopy(pair2[1])
                else:
                    continue
                new_pairs.add((half1, half2))
                if not_put_for_removal:
                    indexes_of_parents_to_remove.append(starting_index + j)
                    not_put_for_removal = False
        return new_pairs, indexes_of_parents_to_remove

    @staticmethod
    def _independent(act, set2, independence_matrix):
        """
        Checks whether a given activity is independent (has "#" relation) of all the activities in a given set.

        Parameters
        ----------
        act: int
            Number-encoded activity.

        set2: frozenset of int
            Set of number-encoded activities.

        independence_matrix: ndarray of bool, shape=[unique_act_num, unique_act_num]
            If for activities i and j independence_matrix[i][j] == True, i and j are independent ('#'),
            if False - dependent ('->', '<-', or '||').

        Returns
        -------
        result: bool
            True if a given activity is independent of all the activities in a given set, False otherwise.

        """
        return np.all(independence_matrix[act][list(set2)])

    @staticmethod
    def create_places_and_edges(graph, places):
        """
        Adds places and edges between places and already created transitions to the graph.

        Parameters
        ----------
        graph: Graph
            Graph.

        places: list of tuple of two sets of str
            List of places. A place is represented by two sets of activities (incoming and outgoing).
        """
        for pair in places:
            set1 = pair[0]
            set2 = pair[1]

            place_id = ','.join(set1) + ' -> ' + ','.join(set2)
            graph.add_node(place_id, '', node_type=NodeType.PLACE)

            for in_node_id in set1:
                graph.add_edge(in_node_id, place_id)
            for out_node_id in set2:
                graph.add_edge(place_id, out_node_id)


def alpha_miner(data_holder: DataHolder, n_jobs: int = 1) -> Graph:
    """
    Realization of an Alpha Miner algorithm.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    n_jobs: int, default=1
        Maximum number of processes created if possible.
        If n_jobs > 0 - max number of processes = n_jobs;
        if n_jobs < 0 - max number of processes = mp.cpu_count() + 1 + n_jobs;
        in other cases: max number of processes = 1.

    Returns
    -------
    graph : Graph

    Notes
    -----
    This algorithm can not handle loops of lengths 1.

    References
    ----------
    W.M.P. van der Aalst, A.J.M.M. Weijters, and L. Maruster. Workflow Mining: Discovering Process Models
    from Event Logs. IEEE Transactions on Knowledge and Data Engineering (TKDE), Accepted for publication, 2003
    """
    miner = AlphaMiner(data_holder, n_jobs)
    miner.apply()
    return miner.graph
