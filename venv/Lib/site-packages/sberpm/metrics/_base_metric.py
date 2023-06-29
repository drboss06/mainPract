import pandas as pd

from ._utils import round_decorator


class BaseMetric:
    """
    Base Class that will contains base metrics for every object.

    Parameters
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    time_unit : {'s'/'second', 'm'/'minute', 'h'/'hour', 'd'/'day', 'w'/'week'}
        Calculate time in needed format.

    Attributes
    ----------
    _dh : DataHolder
        Object that contains the event log and the names of its necessary columns.

    _group_column : str
        Column used for grouping the data.

    _group_data: pandas.GroupBy object
        Object that contains pandas.GroupBy data grouping by _group_column.

    _time_unit: int
        A number time/duration metric values need to be divided by
        so that they transform into needed time format.

    metrics: pd.DataFrame
        DataFrame contain all metrics that can be calculated
    """

    def __init__(self, data_holder, time_unit, round):
        data_holder.check_or_calc_duration()
        self._dh = data_holder
        if time_unit in ('week', 'w'):
            self._time_unit = 604800
        elif time_unit in ('day', 'd'):
            self._time_unit = 86400
        elif time_unit in ('hour', 'h'):
            self._time_unit = 3600
        elif time_unit in ('minute', 'm'):
            self._time_unit = 60
        elif time_unit in ('second', 's'):
            self._time_unit = 1
        else:
            raise ValueError(f'Unknown time unit: "{time_unit}"')

        self.metrics = None
        self._group_column = None
        self._group_data = None
        self._round = round

    def apply(self):
        raise NotImplementedError()

    def calculate_time_metrics(self, std=False):
        """
        Calculates all possible time metrics:
            total_duration
            mean_duration
            median_duration
            min_duration
            max_duration

            variance_duration
            std_duration

        Parameters
        ----------
        std: bool, default=False
            If True, 'variance_duration' and 'std_duration' metrics are also calculated.

        Returns
        -------
        result: pandas.DataFrame
        """
        time_df = pd.DataFrame(index=list(self._group_data.groups.keys())) \
            .join(self.total_duration()) \
            .join(self.mean_duration()) \
            .join(self.median_duration()) \
            .join(self.max_duration()) \
            .join(self.min_duration())

        if std:
            time_df = time_df \
                .join(self.variance_duration()) \
                .join(self.std_duration())

        return time_df

    @round_decorator
    def total_duration(self):
        """
        Return total duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].sum() / self._time_unit).rename('total_duration')

    @round_decorator
    def mean_duration(self):
        """
        Return mean duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].mean() / self._time_unit).rename('mean_duration')

    @round_decorator
    def median_duration(self):
        """
        Return median duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].median() / self._time_unit) \
            .rename('median_duration')

    @round_decorator
    def max_duration(self):
        """
        Return maximum duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].max() / self._time_unit).rename('max_duration')

    @round_decorator
    def min_duration(self):
        """
        Return minimum duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].min() / self._time_unit).rename('min_duration')

    @round_decorator
    def variance_duration(self):
        """
        Return variance of duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].var(ddof=0) / self._time_unit) \
            .rename('variance_duration')

    @round_decorator
    def std_duration(self):
        """
        Return standard deviation of duration.

        Returns
        -------
        result: pandas.Series
        """
        return (self._group_data[self._dh.duration_column].std(ddof=0) / self._time_unit) \
            .rename('std_duration')

    def inclusion_rate(self, selected_activities):
        """
        Return the percentage of ids, containing selected activities for given object.

        Parameters
        -----------
        selected_activities: iterable of str
            List of activities.

        Returns
        -------
        result: pandas.Series
        """
        mask = self._dh.data[self._dh.activity_column].isin(selected_activities)
        selected_ids = set(self._dh.data[self._dh.id_column][mask].unique())

        return (self.unique_ids().apply(lambda x: len(x.intersection(selected_ids))) /
                self.unique_ids_num()).rename('inclusion_rate')

    def calc_metrics(self, *metric_names, raise_no_method=True):
        """
        Calculates the given metrics.
        All the metrics must have the same name as the existing methods od the class.
        If at least one metric does not exist, an error will be raised.

        Parameters
        ----------
        metric_names: iterable of str
            Metric names.

        raise_no_method: bool, default=True
            It true, raise an error if it is impossible to calculate
            at least one of the metrics. Skip 'impossible' metrics otherwise.

        Returns
        -------
        result: pandas.DataFrame
        """
        methods = []
        for name in metric_names:
            method = getattr(self, name, None)
            if method is not None and callable(method):
                methods.append(method)
            else:
                if raise_no_method:
                    raise AttributeError(f"Object '{self.__class__.__name__}' does not have '{name}' method.")

        if len(methods) == 0:
            if raise_no_method:
                raise AttributeError(f"Object '{self.__class__.__name__}' does not have {metric_names} methods.")
            else:
                return pd.DataFrame()

        result = pd.concat([m() for m in methods], axis=1, join='inner')

        return result
