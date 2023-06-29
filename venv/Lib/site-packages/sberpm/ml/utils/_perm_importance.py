import numpy as np
import pandas as pd
from sklearn.metrics import check_scoring
from sklearn.utils import check_array, check_random_state


class PermutationImportance:
    """
    Object that calculates feature importances of the data
    for a given estimator using permutation importance method.

    Parameters
    ----------
    estimator: object
        The base estimator. It needs to be pre-fitted.

    scoring: string, callable or None, default=None
        Scoring function that is used for computing feature importances.
        If ``None``, the ``score`` method of the estimator is used.

    n_iter: int, default=5
        Number of shuffles for each column.

    random_state: integer or numpy.random.RandomState
        Random state.
    """

    def __init__(self, estimator, scoring=None, n_iter=5, random_state=None):
        self.estimator = estimator
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state

        self._scorer = check_scoring(self.estimator, scoring=scoring)
        self._rnd = check_random_state(random_state)

        self.base_score_ = None
        self.feature_importances_ = None
        self.feature_importances_std_ = None

    def fit(self, X, y):
        """
        Calculate feature importances.

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            The training dataset.

        y: array-like, shape [n_samples,]
            The target values.

        Returns
        -------
        self
        """
        # Check and transform the data into np.ndarray
        X = check_array(X)

        base_score = self._scorer(self.estimator, X, y)

        scores_diff = []
        for i in range(self.n_iter):
            scores_shuffled = np.array([self._scorer(self.estimator, X_one_col_shuffled, y)
                                        for X_one_col_shuffled in shuffle_each_column_iterator(X, random=self._rnd)])
            scores_diff.append(base_score - scores_shuffled)

        self.base_score_ = base_score
        self.feature_importances_ = np.mean(scores_diff, axis=0)
        self.feature_importances_std_ = np.std(scores_diff, axis=0)
        return self

    def explain_weights(self, sort_by_weights=True):
        """
        Return a table describing feature importances and their std.

        Parameters
        ----------
        sort_by_weights, bool, default=True
            If True, the table is sorted by feature weights in a descending order,
            otherwise by feature number.
        """
        if self.base_score_ is None:
            raise RuntimeError('Method fit(...) should be called first.')

        df = pd.DataFrame({'feature': [f'x{num}' for num in range(len(self.feature_importances_))],
                           'weight': self.feature_importances_,
                           'std': self.feature_importances_std_})
        if sort_by_weights:
            df = df.sort_values(by='weight', ascending=False).reset_index(drop=True)
        return df


def shuffle_each_column_iterator(X, random):
    """
    Return an iterator of the copies of the data.
    Each copy has values in one of its column shuffled.

    Parameters
    ----------
    X: np.ndarray of shape [n_samples, n_features]
        Data.

    random: np.random.RandomState
        Random state.

    Returns
    -------
    X: np.ndarray of shape [n_samples, n_features]
    """
    X_ = X.copy()

    for column in range(X_.shape[1]):
        random.shuffle(X_[:, column])  # shuffle values in 'column'
        yield X_
        X_[:, column] = X[:, column]  # restore initial values