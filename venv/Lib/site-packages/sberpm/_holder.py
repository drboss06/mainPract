import warnings
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Union, Optional

import pandas as pd
import pandas.api.types as pd_types_utils

from ._utils import generate_data_partitions

warnings.simplefilter('always', UserWarning)


class DataHolder:
    """
        Object that loads, preprocesses and contains the event log and names of its columns.

        Parameters
        ----------
        data: pandas.DataFrame or str
            The event log of the process. It must have three columns: id_column (unique id of a trace),
            activity_column (name of an activity in a trace), dt_column (timestamp of an activity in a particular trace).

            Can be directly a pandas.DataFrame object or a string.
            If a str object is given, it is considered to be a full path to data file.
            Supported file formats are: csv, xls/xlsx, txt.

        id_column: str
            Name of the column that represents the unique id of an event trace.

        activity_column: str
            Name of the column that represents the name of an activity in an event trace.

        start_timestamp_column: str or None, default=None
            Name of the column that represents the start time of an activity.

        end_timestamp_column: str or None, default=None
            Name of the column that represents the end time of an activity.

        user_column: str, default=None
            Name of the column that represents the user that executed a particular activity in a particular event trace.

        text_column  str, default=None
            Name of the column that represents the text comments of a particular activity in a particular event trace.

        duration_column: str or None, default=None
            Name of the column that represents a period of time that was needed to fulfill the activity in a
            particular event trace. The column must be of a numerical type. Values will be converted into seconds.

        duration_unit: {'second', 'minute', 'hour', 'day'}, default=None
            The unit of the duration_column.
            If duration_column is given and it has a numeric format, it is compulsory to set duration_unit as well.

        sep: str, default=','
            Separator between columns. Used only for reading data from file.

        encoding: str, default=None
            Encoding for data. Used only for reading data from file.

        nrows: int, default=None
            Number of rows in data to read. Used only for reading data from file.

        preprocess: bool, default=True
            Whether it is needed to preprocess data.

        time_format: str, default=None
            Time format used for converting string time data to datetime format.
            Examples:
            '13-05-2020 15:30:05' -> '%d-%m-%Y %H:%M:%S'
            '05/13/2020, 15:30' -> '%d/%m/%Y, %H:%M'
            Consult this for time_format syntax:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

        time_errors: {'raise', 'coerce', 'auto_convert'}
            Used only if time_format is not None. Specifies what action to take
            if conversion using time_format has errors (some of the samples
            do not correspont to it).
            If ‘raise’, then invalid parsing will raise an exception.
            If ‘coerce’, then invalid parsing will be set as NaT.
            if 'auto_convert', then auto conversion will be done
            using dayfirst and yearfirst arguments if needed.

        dayfirst: bool, default=None
            Used if time_format is given and time_error is 'auto_convert'
            or if time_format is not given. Used in ambiguous cases only;
            if not given, used as False.
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (True) or month (False). If yearfirst is set to True,
            this distinguishes between YDM and YMD.

        yearfirst: bool, default=None
            Used if time_format is given and  time_error is 'auto_convert'
            or if time_format is not given. Used in ambiguous cases only;
            if not given, used as False.
            Whether to interpret the first value in an ambiguous 3-integer date (e.g. 01/05/09)
            as the year. If True, the first number is taken to be the year,
            otherwise the last number is taken to be the year.

        n_jobs: int, default=1
            If n_jobs > 1, parallel calculation of the data will be used where possible using n_jobs processes.

        Attributes
        ----------
        id_column: str
            Name of the column that represents the unique id of an event trace.

        activity_column: str
            Name of the column that represents the name of an activity in an event trace.

        start_timestamp_column: str or None, default=None
            Name of the column that represents the start time of an activity.

        end_timestamp_column: str or None, default=None
            Name of the column that represents the end time of an activity.

        user_column: str, default=None
            Name of the column that represents the user that executed a particular activity in a particular event trace.

        text_column: str, default=None
            Name of the column that represents the text comments of a particular activity in a particular event trace.

        duration_column: str or None, default=None
            Name of the column that represents a period of time that was needed to fulfill the activity in a
            particular event trace. Values are stored in seconds.

        n_jobs: int, default=1
            If n_jobs > 1, parallel calculation of the data will be used where possible using n_jobs processes.

        data: pd.DataFrame
            The event log data after being preprocessed if "preprocess" parameter is True,
            unpreprocessed data otherwise. This data will be considered to be the event log
            that the algorithms will work with.

        grouped_data: pd.DataFrame
            Returns the event log data grouped by id, with some columns aggregated to tuples.
            Initialized in a lazy way.

        Examples
        --------
        >>> from sberpm import DataHolder
        >>> import pandas as pd
        >>> dh1 = DataHolder('path/to/file.csv', 'id_column', 'activity_column', 'dt_column')
        >>>
        >>> df = pd.DataFrame({
        ... 'id_column': [1, 1, 2],
        ... 'activity_column':['st1', 'st2', 'st1'],
        ... 'dt_column':[123456, 123457, 123458]})
        >>> dh2 = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
        """

    def __init__(self,
                 data: Union[pd.DataFrame, str],
                 id_column: Optional[str],
                 activity_column: str,
                 start_timestamp_column: Optional[str] = None,
                 end_timestamp_column: Optional[str] = None,
                 user_column: Optional[str] = None,
                 text_column: Optional[str] = None,
                 duration_column: Optional[str] = None,
                 duration_unit: Optional[str] = None,
                 sep: str = ',',
                 encoding: Optional[str] = None,
                 nrows: Optional[int] = None,
                 preprocess: bool = True,
                 time_format: Optional[str] = None,
                 time_errors: Optional[str] = 'raise',
                 dayfirst: Optional[bool] = None,
                 yearfirst: Optional[bool] = None,
                 n_jobs: Optional[int] = 1):

        self._id_column = id_column
        self.activity_column = activity_column
        self.start_timestamp_column = start_timestamp_column
        self.end_timestamp_column = end_timestamp_column
        self.user_column = user_column
        self.text_column = text_column
        self.duration_column = duration_column
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count() - n_jobs + 1

        if type(data) == str:
            if data.split('.')[-1] == 'csv':
                raw_data = pd.read_csv(data, sep=sep, encoding=encoding, nrows=nrows)
            elif data.split('.')[-1] in ['xlsx', 'xls']:
                raw_data = pd.read_excel(data, nrows=nrows)
            elif data.split('.')[-1] == 'txt':
                raw_data = pd.read_table(data, sep=sep, encoding=encoding, nrows=nrows)
            else:
                raise ValueError(f"Only 'csv', 'xls(x)' and 'txt' file formats are supported, "
                                 f"but given file path ends with '{data.split('.')[-1]}'")
        elif type(data) == pd.DataFrame:
            raw_data = data
        else:
            raise ValueError(f'pandas.DataFrame or str types are expected, but got {type(data)}')

        if duration_column is not None and pd.api.types.is_numeric_dtype(data[duration_column]):
            if duration_unit is None:
                raise RuntimeError('As long as "duration" column is numeric, '
                                   '"duration_unit" argument must be set in the constructor of DataHolder.')
            elif duration_unit not in {'second', 'minute', 'hour', 'day'}:
                raise ValueError(f"duration_unit should be one of ['second', 'minute', 'hour', 'day'], "
                                 f" got '{duration_unit}' instead.")
            else:
                # Transform to seconds
                if duration_unit == 'second':
                    coeff = 1
                elif duration_unit == 'minute':
                    coeff = 60
                elif duration_unit == 'hour':
                    coeff = 60 * 60
                else:
                    coeff = 60 * 60 * 24
                data[duration_column] = data[duration_column] * coeff

        self.data = \
            self._preprocess_data(raw_data, time_format, time_errors, dayfirst, yearfirst) if preprocess else raw_data
        self.grouped_data = None

    @property
    def id_column(self):
        if self._id_column is not None:
            return self._id_column
        else:
            raise LogWithoutIdError('The event log does not have an ID column. '
                                    'This action is impossible.')

    def _preprocess_data(self, df, time_format, time_errors, dayfirst, yearfirst):
        """
        Does basic preprocessing:
            - remove null values;
            - convert main columns to str types
            - convert timestamps to date_time format (if dt_column given);
            - sort by id and timestamp (if start_timestamp_column or end_timestamp_column are given).

        Parameters
        ----------
        df: pd.DataFrame
            Data for preprocessing.

        time_format: str, default=None
            Time format used for converting string time data to datetime format.
            Examples:
            '13-05-2020 15:30:05' -> '%d-%m-%Y %H:%M:%S'
            '05/13/2020, 15:30' -> '%d/%m/%Y, %H:%M'
            Consult this for time_format syntax:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

        time_errors: {'raise', 'coerce', 'auto_convert'}
            Used only if time_format is not None. Specifies what action to take
            if conversion using time_format has errors (some of the samples
            do not correspont to it).
            If ‘raise’, then invalid parsing will raise an exception.
            If ‘coerce’, then invalid parsing will be set as NaT.
            if 'auto_convert', then auto conversion will be done
            using dayfirst and yearfirst arguments if needed.

        dayfirst: bool, default=None
            Used if time_format is given and time_error is 'auto_convert'
            or if time_format is not given. Used in ambiguous cases only;
            if not given, used as False.
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (True) or month (False). If yearfirst is set to True,
            this distinguishes between YDM and YMD.

        yearfirst: bool, default=None
            Used if time_format is given and  time_error is 'auto_convert'
            or if time_format is not given. Used in ambiguous cases only;
            if not given, used as False.
            Whether to interpret the first value in an ambiguous 3-integer date (e.g. 01/05/09)
            as the year. If True, the first number is taken to be the year,
            otherwise the last number is taken to be the year.

        Returns
        -------
        df: pd.DataFrame
            Preprocessed data.
        """
        # Drop null rows having None values in id_column
        if self._id_column is not None:
            full_mask = None
            for col in [self._id_column]:
                if col is not None:
                    mask = df[col].isna()
                    if mask.sum() > 0:
                        warnings.warn(f'DataHolder: column {col} has {mask.sum()} None values, '
                                      f'the corresponding rows will be removed.', UserWarning)
                    full_mask = mask if full_mask is None else full_mask | mask
            df = df.drop(df.index[full_mask], axis=0)

        # Convert columns to string
        for col in [self._id_column, self.activity_column, self.user_column, self.text_column]:
            if col is not None and not pd_types_utils.is_string_dtype(col):
                df[col] = df[col].astype(str)

        # Convert time to pd.date_time
        for time_column in [self.start_timestamp_column, self.end_timestamp_column]:
            if time_column is not None and not pd_types_utils.is_datetime64_any_dtype(time_column):
                if time_format is None:
                    warnings.warn("DataHolder: 'time_format' is not set, "
                                  "recommended to specify it for correct time conversion, "
                                  "e.g., time_format='%d-%m-%Y %H:%M:%S'")
                    dayfirst, yearfirst = DataHolder._check_dayfirst_yearfirst(dayfirst, yearfirst)
                    df[time_column] = pd.to_datetime(df[time_column], dayfirst=dayfirst, yearfirst=yearfirst, utc=True)

                elif time_errors == 'raise':
                    df[time_column] = pd.to_datetime(df[time_column], format=time_format, errors='raise', utc=True)
                elif time_errors == 'coerce':
                    df[time_column] = pd.to_datetime(df[time_column], format=time_format, errors='coerce', utc=True)
                elif time_errors == 'auto_convert':
                    dayfirst, yearfirst = DataHolder._check_dayfirst_yearfirst(dayfirst, yearfirst)
                    result = pd.to_datetime(df[time_column], format=time_format, errors='coerce', utc=True)
                    na_mask = result.isna()
                    non_converted_timestamps = df[time_column][na_mask]
                    res = pd \
                        .to_datetime(non_converted_timestamps, dayfirst=dayfirst, yearfirst=yearfirst, utc=True)
                    df[time_column] = result
                    df.loc[na_mask, time_column] = res
                else:
                    raise ValueError("time_errors must be in "
                                     f"['raise', 'coerce', 'auto_convert'], but got '{time_errors}' instead")
        # Sort
        if self._id_column is not None:
            timestamp_col = self.get_timestamp_col()
            if timestamp_col is None:  # no timestamp columns
                df = df.sort_values(self.id_column)
                warnings.warn('DataHolder: time column is not given, cannot sort the activities.', UserWarning)
            elif self.is_interval():  # 2 timestamp columns
                df = df.sort_values([self.id_column, self.start_timestamp_column, self.end_timestamp_column])
            else:  # one timestamp column
                df = df.sort_values([self.id_column, timestamp_col])
        else:  # id_column is None
            timestamp_col = self.get_timestamp_col()
            if timestamp_col is None:  # no timestamp columns
                raise RuntimeError('The log that does not have id and any timestamp columns cannot be processed.')
            elif self.is_interval():  # 2 timestamp columns
                df = df.sort_values([self.start_timestamp_column, self.end_timestamp_column])
            else:  # one timestamp column
                df = df.sort_values([timestamp_col])

        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _check_dayfirst_yearfirst(dayfirst, yearfirst):
        if dayfirst is None:
            DataHolder._warn_not_set('dayfirst')
            dayfirst = False
        if yearfirst is None:
            DataHolder._warn_not_set('yearfirst')
            yearfirst = False
        return dayfirst, yearfirst

    @staticmethod
    def _warn_not_set(argument):
        warnings.warn(f"DataHolder: timestamp auto conversion will be done. '{argument}' "
                      "is not set, in ambiguous cases it will be considered as False.", UserWarning)

    def get_grouped_data(self, *columns):
        """
        Returns data grouped by id, with given columns aggregated to tuples.

        Parameters
        ----------
        columns: array-like of str
            Columns for aggregating.

        Returns
        -------
        grouped_data: pd.DataFrame
            Data grouped by id, with given columns aggregated to tuples. The id_column is a separate column.
        """
        columns_to_agg = columns if self.grouped_data is None else \
            [col for col in columns if col not in self.grouped_data.columns]
        if len(columns_to_agg) != 0:
            # Group the data
            if self.n_jobs == 1:
                grouped_result = self._groupby(self.data, self.id_column, *columns_to_agg)
            else:
                pool = Pool(self.n_jobs)
                result_objects = [pool.apply_async(self._groupby,
                                                   args=(sub_data, self.id_column, *columns_to_agg)) for sub_data in
                                  generate_data_partitions(self.data, self.id_column, batch_num=self.n_jobs * 2)]
                grouped_result = pd.concat([r.get() for r in result_objects])
                pool.close()
                pool.join()
            # Save the result
            if self.grouped_data is None:
                self.grouped_data = grouped_result.reset_index()  # make 'id' a column (not an index)
            else:
                self.grouped_data = self.grouped_data.join(grouped_result, on=self.id_column, how='inner')

        return self.grouped_data[[self.id_column, *columns]]

    def get_grouped_columns(self, *columns):
        """
        Returns separate columns (Series) of the data aggregated by given columns.

        Parameters
        ----------
        columns: array-like of str
            Columns for aggregating.

        Returns
        -------
        result: pd.Series or generator of pd.Series
            One or several separate columns of aggregated data.
        """
        grouped_data = self.get_grouped_data(*columns)
        if len(columns) == 1:
            return grouped_data[columns[0]]
        else:
            return (grouped_data[col] for col in columns)

    def get_unique_activities(self):
        """
        Returns unique activities in the event log.

        Returns
        -------
        activities: array-like of str
            Names of unique activities.
        """
        return self.data[self.activity_column].unique()

    def get_columns(self):
        """
        Returns columns of the data.

        Returns
        -------
        columns: array-like of str
            Columns' names.
        """
        return self.data.columns

    def get_text(self):
        """
        Returns text column of the data or None, if text_column was not given in the constructor.

        Returns
        -------
        text_data: array-like of str
            Column with text data.
        """
        return self.data[self.text_column] if self.text_column is not None else None

    @staticmethod
    def _groupby(data, groupby_column, *agg_columns):
        """
        Groups given data by id and aggregates all given columns to tuples.

        Parameters
        ----------
        data: pd.DataFrame
            Data.

        groupby_column: str
            Column for grouping the data.

        agg_columns: array-like of str
            Columns for aggregating to tuples.

        Returns
        -------
        grouped_data: pd.DataFrame
            Data grouped by id, with columns aggregated to tuples. Index: id.
        """
        return data.groupby(groupby_column, as_index=True).agg({col: tuple for col in agg_columns})

    def check_or_calc_duration(self):
        """
        Calculates duration if it is not calculated.
        """
        if self.duration_column is None:
            df = self.data
            id_column = self.id_column
            start_timestamp_column = self.start_timestamp_column
            end_timestamp_column = self.end_timestamp_column
            duration_column = 'duration'
            if start_timestamp_column is None and end_timestamp_column is None:
                raise RuntimeError('Cannot calculate time difference, '
                                   'because both "start_timestamp_column" and "end_timestamp_column" are None.')
            elif start_timestamp_column is not None and end_timestamp_column is not None:
                df[duration_column] = df[end_timestamp_column] - df[start_timestamp_column]
            else:
                if self.start_timestamp_column is not None and self.end_timestamp_column is None:
                    start_timestamp_col = df[start_timestamp_column]
                    end_timestamp_col = df[start_timestamp_column].shift(-1)
                    start_id_col = df[id_column]
                    end_id_col = df[id_column].shift(-1)
                else:
                    start_timestamp_col = df[end_timestamp_column].shift(1)
                    end_timestamp_col = df[end_timestamp_column]
                    start_id_col = df[id_column].shift(1)
                    end_id_col = df[id_column]
                df[duration_column] = end_timestamp_col - start_timestamp_col
                different_id_mask = start_id_col != end_id_col
                df.loc[different_id_mask, duration_column] = None

            df[duration_column] = df[duration_column] / pd.Timedelta(seconds=1)
            self.data = df
            self.duration_column = duration_column

    def get_timestamp_col(self) -> Union[str, None]:
        """
        Returns one non-none timestamp column name.
        If all timestamp columns are None, returns None.

        Returns
        -------
        col: str or None
        """
        col = self.start_timestamp_column
        if col is None:
            col = self.end_timestamp_column
        return col

    def is_interval(self) -> bool:
        """
        Check whether is it an interval log
        (has timestamps both of the beginning and ending of activities).

        Returns
        -------
        is_interval: bool
        """
        return self.start_timestamp_column is not None and \
               self.end_timestamp_column is not None

    def top_traces_dh(self, n: int) -> 'DataHolder':
        """
        Return DataHolder with the data of n most frequent traces.

        Parameters
        ----------
        n: int
            Number of top frequent traces.

        Returns
        -------
        new_data_holder: DataHolder
        """
        # copy data_holder without dataframes
        data, grouped_data = self.data, self.grouped_data
        self.data, self.grouped_data = None, None
        new_dh = deepcopy(self)
        self.data, self.grouped_data = data, grouped_data

        dfg = self.get_grouped_data(self.activity_column)  # chains of activities
        id_series = dfg.groupby(self.activity_column) \
            .agg({self.id_column: (tuple, len)})[self.id_column] \
            .sort_values('len', ascending=False)['tuple'].head(n)
        top_ids = set(id_ for id_tuples in id_series for id_ in id_tuples)
        new_dh.data = self.data[self.data[self.id_column].isin(top_ids)]

        return new_dh

    def copy(self):
        """
        Return new DataHolder with the data of n most frequent traces.

        Returns
        -------
        new_data_holder: DataHolder
        """
        return deepcopy(self)


class LogWithoutIdError(Exception):
    pass
