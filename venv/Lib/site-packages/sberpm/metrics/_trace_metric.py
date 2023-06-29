import pandas as pd

from ._base_metric import BaseMetric
from ._utils import round_decorator


class TraceMetric(BaseMetric):
    """
    Class for calculating metrics for event traces.

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
        DataFrame that contains all calculated metrics.
    """

    def __init__(self, data_holder, time_unit='hour', round=None):
        super().__init__(data_holder, time_unit, round)

        cols_to_aggregate = [data_holder.activity_column]
        if data_holder.user_column is not None:
            cols_to_aggregate.append(data_holder.user_column)
        self._grouped_data = data_holder.get_grouped_data(*cols_to_aggregate)  # id_column and traces

        duration_df = data_holder.data.groupby(data_holder.id_column)[data_holder.duration_column].sum()
        self._grouped_data = self._grouped_data.join(duration_df, on=data_holder.id_column)

        self._group_column = data_holder.activity_column
        self._group_data = self._grouped_data.groupby(self._group_column)
        self._traces = pd.DataFrame({self._dh.activity_column: self._grouped_data[self._dh.activity_column].unique()}) \
            .set_index(self._dh.activity_column, drop=False)[self._dh.activity_column]  # pandas.Series

    def apply(self):
        """
        Calculate all possible metrics for this object.

        Returns
        -------
        result: pandas.DataFrame
        """
        self.metrics = pd.DataFrame(index=self._grouped_data[self._dh.activity_column].unique()) \
            .join(self.count()) \
            .join(self.ids()) \
            .join(self.trace_length()) \
            .join(self.unique_activities_num()) \
            .join(self.loop_percent())

        if self._dh.user_column is not None:
            self.metrics = self.metrics \
                .join(self.unique_users()) \
                .join(self.unique_users_num())

        self.metrics = self.metrics.join(self.calculate_time_metrics(True))

        return self.metrics.sort_values('count', ascending=False)

    def count(self):
        """
        Return number of occurrences of the trace in the event log.
        (=number of IDs that have the given trace).

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.id_column].count().rename('count')  # or .nunique() - no difference

    def ids(self):
        """
        Return list of IDs that have the given trace (=sequence of activities).

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data.agg({self._dh.id_column: set})[self._dh.id_column].rename('ids')

    def trace_length(self):
        """
        Return length of the trace.

        Returns
        -------
        result: pandas.Series
        """
        return self._traces.apply(len).rename('trace_length')

    def unique_activities(self):
        """
        Return unique activities of the trace.

        Returns
        -------
        result: pandas.Series
        """
        return self._traces.apply(lambda x: set(x)).rename('unique_activities')

    def unique_activities_num(self):
        """
        Return number of unique activities of the trace.

        Returns
        -------
        result: pandas.Series
        """
        return self._traces.apply(lambda x: len(set(x))).rename('unique_activities_num')

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
        Return set of unique users who worked on the IDs that have given event trace
        (=sequence of activities).

        Returns
        -------
        result: pandas.Series
        """
        return self._group_data[self._dh.user_column].apply(lambda x: set().union(*x)).rename('unique_users')

    def unique_users_num(self):
        """
        Return number of unique users who worked on the IDs that have given event trace
        (=sequence of activities).

        Returns
        -------
        result: pandas.Series
        """
        return self.unique_users().apply(len).rename('unique_users_num')
