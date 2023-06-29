import re
import warnings
from collections import Counter
from typing import Optional, List

import numpy as np
import pandas as pd
import sqldf
from sklearn import svm
from sklearn.cluster import DBSCAN

from ..._holder import DataHolder

warnings.filterwarnings('ignore')


class Chronometrage:
    def __init__(self, data_holder: DataHolder,
                 query: Optional[str] = None,
                 start_query: Optional[str] = None,
                 end_query: Optional[str] = None,
                 change_columns: List[str] = None,
                 sort_params: Optional[List[str]] = None
                 ) -> None:
        """
        Parameters query, start_query, end_query can be of "sql" or "pandas" type,
        they both must refer to the dataframe as "df",
        they should return one column: a boolean mask or a column of 0 and 1.
        A "True" or "1" means:
            - the start of a new process for "start_query",
            - the end of the process for "end_query",
            - the start of a new process on this line or finish of the process on the previous line for "query".

        In case of "sql" type they must look like "SELECT ... from df".
        """

        # check params
        if not ((query is not None and start_query is None and end_query is None) or
                (query is None and (start_query is not None or end_query is not None))):
            raise ValueError(
                'Either only "query" or at least one of the "start_query"/"end_query" parameters must be given.')

        self.query_types = {q: 'sql' if is_sql_query(q) else 'pandas' for q in
                            [q for q in [query, start_query, end_query] if q is not None]}
        self.data = data_holder.data.copy()  # so that data in dh is not changed
        self.time_column = data_holder.get_timestamp_col()
        self.query = query
        self.start_query = start_query
        self.end_query = end_query
        self.change_columns = [] if change_columns is None else change_columns

        self.sort_params = sort_params

    def sort_by(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(self.sort_params).reset_index(drop=True) if self.sort_params is not None else df

    def create_pro_n(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates 'PRO_N' column.
        """
        if self.query is not None:
            mask = self.run_query(df, self.query)
        else:
            mask = pd.Series(np.full(df.shape[0], False))
            if self.start_query is not None:
                mask |= self.run_query(df, self.start_query)
            if self.end_query is not None:
                mask |= self.run_query(df, self.end_query).shift(1).fillna(False)
            for col in self.change_columns:
                mask |= df[col] != df[col].shift(1)

        df['PRO_N'] = mask.astype(int).cumsum()
        return df

    def add_event_date_start(self, df: pd.DataFrame) -> pd.DataFrame:
        df['event_date_start_'] = df[self.time_column].shift(1)
        first_date_series = df.groupby('PRO_N')['event_date_start_'].min().rename('event_date_start')
        df = df.join(first_date_series, on='PRO_N', how='left').drop('event_date_start_', axis=1)
        return df

    def get_chrono(self):
        df = self.create_pro_n(self.sort_by(self.data))  # check if there is pro_n column
        df = self.add_event_date_start(df)  # getting true event start date

        self.df = df

        time_duration = df.groupby(['PRO_N'])[self.time_column].agg(['max', 'min'])
        time_duration = (time_duration['max'] - time_duration['min']) / np.timedelta64(1, 's')
        time_duration = np.array(time_duration).reshape(-1, 1)
        result = algo_anomaly(time_duration)

        if result is None:
            return {}

        drop = {'average time': result[0],
                'number of selected items': result[1],
                'unique change items': len(self.data.groupby(self.change_columns).size()),
                'max unique id': df['PRO_N'].nunique()}
        return drop

    def run_query(self, df, query) -> pd.Series:
        """
        Runs query and returns a pd.Series of type bool.
        """
        if self.query_types[query] == 'sql':
            result = sqldf.run(query)
            if len(result.columns) != 1:
                raise ValueError(
                    f'Query "{query}" must return one column, but returns {len(result.columns)}.')
            result = result[result.columns[0]]
        else:  # pandas
            result = eval(query)
        return result.astype(bool)


def algo_anomaly(X: np.ndarray):
    try:
        best_params_svm = {'cache_size': 200, 'coef0': 0.0, 'degree': 4, 'gamma': 'scale', 'kernel': 'poly',
                           'max_iter': -1, 'nu': 0.3, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        best_params_db_scan = {'algorithm': 'auto', 'eps': 0.1, 'leaf_size': 30, 'metric': 'euclidean',
                               'metric_params': None, 'min_samples': 9, 'n_jobs': None, 'p': None}

        left = svm.OneClassSVM().set_params(**best_params_svm)
        labels_svm = left.fit(X).predict(X)
        labels_svm = np.array([int((-t + 1) / 2) for t in labels_svm])

        right = DBSCAN().set_params(**best_params_db_scan)
        labels_dbscan = right.fit_predict(X)

        if len(set(labels_svm)) == 1:
            labels_svm = np.array([0] * len(labels_svm))
        if len(set(labels_dbscan)) == 1:
            if labels_dbscan[0] != 0:
                labels_dbscan = labels_dbscan / labels_dbscan[0] - 1
        else:
            idx = list(X).index(max(X))
            if Counter(labels_dbscan).most_common()[0][0] != labels_dbscan[idx]:
                idx = list(X).index(max(X))
                labels_dbscan = np.array([1 if x == idx else 0 for x in labels_dbscan])
            else:
                labels_dbscan = np.array([0 if x == idx else 1 for x in labels_dbscan])

        labels = [labels_svm[i] or labels_dbscan[i] for i in range(len(labels_svm))]

        selected = [X[i] for i in range(len(X)) if labels[i] == 0]
        return np.mean(np.array(selected)), len(selected)
    except ValueError:
        return None


def is_sql_query(query: str) -> bool:
    pattern = re.compile("^SELECT\s.*\sFROM\sdf$", re.IGNORECASE)
    return pattern.match(query) is not None
