import pandas as pd

from ._base_metric import BaseMetric
from ._utils import round_decorator


class TransitionMetric(BaseMetric):
    """
    Class for calculating metrics for transitions
    (pairs of consequent activities).

    Parameters
    ----------
    data_holder: DataHolder
        Object that contains the event log and the names of its necessary columns.

    time_unit: {'s'/'second', 'm'/'minute', 'h'/'hour', 'd'/'day', 'w'/'week'}, default='hour'
        Calculate time/duration values in given format.

    round: int, default=None
        Round float values of the metrics to the given number of decimals.


    Attributes
    ----------
    metrics: pd.DataFrame
        DataFrame that contains all calculated metrics.
    """

    def __init__(self, data_holder, time_unit='hour', round=None):
        super().__init__(data_holder, time_unit, round)

        act1 = data_holder.data[data_holder.activity_column]
        act2 = data_holder.data[data_holder.activity_column].shift(-1)
        id1 = data_holder.data[data_holder.id_column]
        id2 = data_holder.data[data_holder.id_column].shift(-1)
        tr_data = data_holder.data

        self._group_column = 'transition'
        tr_data[self._group_column] = list(zip(act1, act2))
        self._tr_data = tr_data[id1 == id2]

        self._group_data = self._tr_data.groupby(self._group_column)

    def apply(self):
        """
        Calculate all possible metrics for this object.

        Returns
        -------
        result: pandas.DataFrame
        """
        self.metrics = pd.DataFrame(index=self._tr_data[self._group_column].unique()) \
            .join(self.count()) \
            .join(self.unique_ids()) \
            .join(self.unique_ids_num()) \
            .join(self.aver_count_in_trace()) \
            .join(self.loop_percent()) \
            .join(self.throughput())

        if self._dh.user_column is not None:
            self.metrics = self.metrics \
                .join(self.unique_users()) \
                .join(self.unique_users_num())

        self.metrics = self.metrics.join(self.calculate_time_metrics(True))

        return self.metrics.sort_values('count', ascending=False)

    def count(self):
        """
        Returns the number of occurrences of transitions in the event log.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._group_column].count().rename('count')

    def unique_ids(self):
        """
        Return list of unique IDs that have the given transition.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data.agg({self._dh.id_column: set})[self._dh.id_column].rename('unique_ids')

    def unique_ids_num(self):
        """
        Return number of unique IDs that have the given transition.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.id_column].nunique().rename('unique_ids_num')

    @round_decorator
    def aver_count_in_trace(self):
        """
        Return average count of a transition in those event traces
        where the transition occurred:

         = total_count_of_transition / num_of_unique_ids_with_this_transition.

        Thus, the minimum possible value is 1 (when activity occurs in ids exactly once).

        Returns
        -------
        result: pandas.Series
        """
        return (self.count() / self.unique_ids_num()).rename('aver_count_in_trace')

    @round_decorator
    def loop_percent(self):
        """
        Return the percentage of transitions that occurred for the 2nd, 3rd, 4th,...
        time in the event traces (percentage of 'extra use' of transitions):

         = (1 - num_of_unique_ids_with_this_transition / total_count_of_transition) * 100.

        Thus, this value ranges from 0 to 1 (non-including) with zero being the best value.

        Returns
        -------
        result: pandas.Series

        """
        return ((1 - self.unique_ids_num() / self.count()) * 100).rename('loop_percent')

    def unique_users(self):
        """
        Return set of unique users that worked on given transition.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data.agg({self._dh.user_column: set})[self._dh.user_column].rename('unique_users')

    def unique_users_num(self):
        """
        Return number of unique users that worked on given transition.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.user_column].nunique().rename('unique_users_num')

    @round_decorator
    def throughput(self):
        """
        Return the average number of times the given transition is performed per time unit.

        Returns
        -------
        result: pandas.Series
        """
        return (self.count() / self.total_duration()).rename('throughput')

    @round_decorator
    def success_rate(self, success_activities):
        """
        Return the percentage of successful ids for given transition.

        Parameters
        -----------
        success_activities: iterable of str
            List of activities.

        Returns
        -------
        result: pandas.Series
        """
        return self.inclusion_rate(success_activities).rename('success_rate')

    @round_decorator
    def failure_rate(self, failure_activities):
        """
        Return the percentage of failed ids for given transition.

        Parameters
        -----------
        failure_activities: iterable of str
            List of activities.

        Returns
        -------
        result: pandas.Series
        """
        return self.inclusion_rate(failure_activities).rename('failure_rate')
