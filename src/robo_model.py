
# -------------------------------------------------------------------------------
# Name:        Robo Estimator Model
# Purpose:     Wrapper for sklearn Regularized Logistic Regression
#
#
# Author:      Mohammad Maysami
#
# Created:     June 2019
# Copyright:   (c) MM 2019
# Licence:     See Git
# -------------------------------------------------------------------------------

from abc import abstractmethod, ABCMeta
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# -------------------------------------------------------------------------------
# Name:        Robo Estimator Model
# Purpose:     Wrapper for sklearn Regularized Logistic Regression
#
#
# Author:      Mohammad Maysami
#
# Created:     June 2019
# Copyright:   (c) MM 2019
# Licence:     See Git
# -------------------------------------------------------------------------------

# from abc import abstractmethod, ABCMeta
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss


class RoboLogistic(LogisticRegression):
    """
    Logistic Regression Class

    Assumptions: All input features will be given as pandas DataFrames with a mix of numeric and categorical data.
            X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
            y = np.array([0, 0, 1])


    """
    # __metaclass__  = ABCMeta

    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """
        Initialize Class Instance

        :param args:   Arguments to be passed for initializing Parent Class (LogisticRegression)
        :param kwargs: Keyword Arguments to be passed for initializing Parent Class (LogisticRegression)
        """

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Fit on training data

        :param X: pd.DataFrame, Input features
        :param y: np.ndarray, Ground truth labels as a numpy array of 0-s and 1-s.
        :return: None, Update/Fit instance of model class
        """

        super().fit(X, y)
        return self

    # ------------------------------------------------------------------
    def process(self, X, y):

        return None

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels on new data

        :param X: pd.DataFrame, Input features
        :param y:
        :return: np.ndarray
                    np.array([1, 0, 1])
        """

        yh = super().predict(X)
        return yh

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Predict the probability of each label

        :param X: pd.DataFrame, Input features
        :param y:
        :return: np.ndarray
                    np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        """

        p = super().predict_proba(X)
        return p

    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        """
        Get the value of the  metrics: F1-score, LogLoss

        :param X: pd.DataFrame, Input features
        :param y: np.ndarray, Ground truth labels as a numpy array of 0-s and 1-s.
        :return: dict
                    {'f1_score': 0.3, 'logloss': 0.7}
        """

        sc_f1 = f1_score(y, self.predict(X))
        sc_log = log_loss(y, self.predict_proba(X))

        return {'f1_score':sc_f1, 'logloss':sc_log}

    # ------------------------------------------------------------------
    def tune_parameters(self, X, y=None):
        """
        Run K-fold cross validation to choose the best parameters

        Note: Output the average scores across all CV validation partitions and best parameters

        :param X: pd.DataFrame, Input features
        :param y: np.ndarray, Ground truth labels as a numpy array of 0-s and 1-s.
        :return:  dict
                    {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag', ‘scores’:
                     {'f1_score': 0.3, 'logloss': 0.7}}
        """

        return {}