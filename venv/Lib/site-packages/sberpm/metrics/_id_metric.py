import pandas as pd

from ._base_metric import BaseMetric
from ._utils import round_decorator


class IdMetric(BaseMetric):
    """
    Class for calculating metrics for IDs.

    Parameters
    ----------
    data_holder: DataHolder
        Object that contains the event log and the names of its necessary columns.

    time_unit: {'s'/'second', 'm'/'minute', 'h'/'hour', 'd'/'day', 'w'/'week'}, default='day'
        Calculate time/duration values in given format.

    round: int, default=None
        Round float values of the metrics to the given number of decimals.


    Attributes
    ----------
    metrics: pd.DataFrame
        DataFrame that contains calculated metrics.
    """

    def __init__(self, data_holder, time_unit='hour', round=None):
        super().__init__(data_holder, time_unit, round)
        self._group_column = data_holder.id_column
        self._group_data = self._dh.data.groupby(self._group_column)

    def apply(self):
        """
        Calculate all possible metrics for this object.

        Returns
        -------
        result: pandas.DataFrame
        """
        self.metrics = pd.DataFrame(index=self._dh.data[self._group_column].unique()) \
            .join(self.trace()) \
            .join(self.trace_length()) \
            .join(self.unique_activities()) \
            .join(self.unique_activities_num()) \
            .join(self.loop_percent())

        if self._dh.user_column is not None:
            self.metrics = self.metrics\
                .join(self.unique_users())\
                .join(self.unique_users_num())

        self.metrics = self.metrics.join(self.calculate_time_metrics(True))

        return self.metrics

    def trace(self):
        """
        Return traces corresponding to IDs.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data.agg({self._dh.activity_column: tuple})[self._dh.activity_column] \
            .rename('trace')

    def trace_length(self):
        """
        Return lengths of traces corresponding to IDs.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.activity_column].count().rename('trace_length')

    def unique_activities(self):
        """
        Return unique activities in traces corresponding to IDs.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data.agg({self._dh.activity_column: set})[self._dh.activity_column] \
            .rename('unique_activities')

    def unique_activities_num(self):
        """
        Return number of unique activities in traces corresponding to IDs.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.activity_column].nunique().rename('unique_activities_num')

    @round_decorator
    def loop_percent(self):
        """
        Return the percentage of activities in the event trace that occurred
        for the 2nd, 3rd, 4th,... time (percentage of 'extra use' of the activities):

         = (1 - num_of_unique_activities / trace_length) * 100.

        Thus, this value ranges from 0 to 1 (non-including).

        Returns
        -------
        result: pandas.Series
        """
        return ((1 - self.unique_activities_num() / self.trace_length()) * 100).rename('loop_percent')

    def unique_users(self):
        """
        Return number of unique users who worked on the given ID.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data.agg({self._dh.user_column: set})[self._dh.user_column].rename('unique_users')

    def unique_users_num(self):
        """
        Return unique users who worked on the given ID.

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.user_column].nunique().rename('unique_users_num')
