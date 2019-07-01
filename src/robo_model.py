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
from functools import wraps
from inspect import getfullargspec
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted  # , check_consistent_length, check_X_y, check_array
from sklearn.metrics import make_scorer, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# TODO: add pipeline if applicable
# from sklearn.pipeline import Pipeline

from src.robo_prep import RoboFeaturizer


# =============================================================
# Generalize Decorator of Function with Arguments
# =============================================================
def robo_preprocess(variable='X'):  # , preprocess=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # class methods have self as args[0]
            preprocess = args[0].preprocess
            if callable(preprocess):
                raise (AttributeError, "Function decorator used without callabe preprocess argument for %s!" % func.__name__)
            # Get list of function variables as
            # (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)
            _inspect_args = getfullargspec(func)

            # Check if variable in function variables
            if variable in _inspect_args.args:

                if variable in kwargs:
                    # variable pass as named argument e.g. func(..., var=value, ...)
                    kwargs[variable] = preprocess.fit_transform(kwargs[variable])
                else:
                    # variable pass as positional value e.g. func(..., value, ...)

                    args = list(args)
                    pos = 0
                    for i in range(_inspect_args.args.index(variable)):
                        if _inspect_args.args[i] not in kwargs:
                            # Function argument passed as value without name
                            pos += 1
                    args[pos] = preprocess.fit_transform(args[pos])
            else:
                raise Warning("Function decorator parameter %s NOT Found in function %s arguments!" % (variable, func.__name__))
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ======================================================================
class RoboLogistic(LogisticRegression, BaseEstimator, ClassifierMixin):
    """
    Logistic Regression Class

    Assumptions: All input features will be given as pandas DataFrames with a mix of numeric and categorical data.
            X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
            y = np.array([0, 0, 1])


    Note: All estimators should specify all the parameters in  __init__ as explicit keyword arguments
        (no *args or **kwargs).

    :param max_unique_for_discrete: [int], Encode columns of up to this threshold of unique numeric values, using one-hot-encoding.
    :param max_missing_to_keep: [0=<float<=1], Drop columns if fraction (%) of missing values are higher than this threshold
    :param add_missing_flag: [Bool], For columns with missing values smaller than the threshold, add a binary column to mark missing data points.
    :param encode_categorical: [Bool], Encode categorical (non-numerical) columns
    :param max_category_for_ohe: [int], Encode categorical columns with number of categories up to this threshold,
                use one-hot-encoding and for the rest use alternate. Only effective if `encode_categorical=True`
    :param scaler: [None or sklearn scaler]:  No scaling of input if None, scale input using 'fit_transform' method

    :params LogisticRegression Class parameters
        {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True,
        'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'warn', 'n_jobs': None,
        'penalty': 'l2', 'random_state': None, 'solver': 'warn', 'tol': 0.0001,
        'verbose': 0, 'warm_start': False}
    """

    # __metaclass__  = ABCMeta
    # ------------------------------------------------------------------
    def __init__(self,
                 max_unique_for_discrete=10,
                 max_missing_to_keep=0.80,
                 add_missing_flag=False,
                 encode_categorical=True,
                 max_category_for_ohe=10,
                 scaler=StandardScaler(),
                 C=1.0, class_weight=None, dual=False,
                 fit_intercept=True, intercept_scaling=1,
                 max_iter=100, multi_class='warn',
                 n_jobs=None, penalty='l2', random_state=None, solver='warn',
                 tol=0.0001, verbose=0, warm_start=False):

        # # Validate Initialization Parameters (Only additional ones to LogisticRegression)
        assert max_unique_for_discrete >= 0, "max_unique_for_discrete must be non-negative integer."
        assert 0 <= max_missing_to_keep <= 1, "max_missing_to_keep must be between 0 and 1."
        assert isinstance(add_missing_flag, bool), "add_missing_flag must be boolean."
        assert isinstance(encode_categorical, bool), "encode_categorical must be boolean."
        assert max_category_for_ohe >= 0, "max_category_for_ohe must be none-negative integer."
        assert (scaler is None) or isinstance(scaler, (Normalizer, StandardScaler, MinMaxScaler)), \
            "scaler must be either None or one of predefined sklearn Scalers"

        self.max_unique_for_discrete = max_unique_for_discrete
        self.max_missing_to_keep = max_missing_to_keep
        self.add_missing_flag = add_missing_flag
        self.encode_categorical = encode_categorical
        self.max_category_for_ohe = max_category_for_ohe
        self.scaler = scaler

        # TODO: Might be helpful to add Feature Selection in preprocessing
        self.preprocess = RoboFeaturizer(max_unique_for_discrete=max_unique_for_discrete,
                                         max_missing_to_keep=max_missing_to_keep,
                                         add_missing_flag=add_missing_flag,
                                         encode_categorical=encode_categorical,
                                         max_category_for_ohe=max_category_for_ohe,
                                         scaler=scaler)

        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                         class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter,
                         multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        # Set range of hyper-parameter set for tuning (CVGridSearch)
        self.hypeparam_grid = {'penalty': ['l2'],
                               'tol': [1e-6, 1e-5, 1e-4, 1e-3],
                               'C': [10 ** p for p in range(-2, 2)],
                               # 'fit_intercept': [True, False],
                               'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                               'scaler': [None, StandardScaler()]
                               }

        # # Smaller hyper-param Grid
        # self.hypeparam_grid = {
        #     'C': [10 ** p for p in range(-1, 1)],
        #     'tol': [1e-5, 1e-4, 1e-3],
        #     # 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        #     # 'scaler': [None, StandardScaler()]
        # }

    # ------------------------------------------------------------------
    @robo_preprocess('X')
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Fit on training data

        :param X: pd.DataFrame, Input features
        :param y: np.ndarray, Ground truth labels as a numpy array of 0-s and 1-s.
        :return: None, Update/Fit instance of model class
        """

        # Using decorator instead for data pre-processing
        # X = self.preprocess.transform(X)

        super().fit(X, y)
        return None

    # ------------------------------------------------------------------
    @robo_preprocess('X')
    def predict(self, X):
        """
        Predict class labels on new data

        :param X: pd.DataFrame, Input features
        :return: np.ndarray
                    e.g. np.array([1, 0, 1])
        """

        # Check is fit had been called
        check_is_fitted(self, ['classes_', 'coef_'])

        # Using decorator instead for data pre-processing
        #  X = self.preprocess.transform(X)

        return super().predict(X)

    # ------------------------------------------------------------------
    @robo_preprocess('X')
    def predict_proba(self, X):
        """
        Predict the probability of each label

        :param X: pd.DataFrame, Input features
        :return: np.ndarray
                    np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        """
        # Check is fit had been called
        check_is_fitted(self, ['classes_', 'coef_'])

        # Using decorator instead for data pre-processing
        # X = self.preprocess.transform(X)

        return super().predict_proba(X)

    # ------------------------------------------------------------------
    @robo_preprocess('X')
    def evaluate(self, X, y):
        """
        Get the value of the  metrics: F1-score, LogLoss

        :param X: pd.DataFrame, Input features
        :param y: np.ndarray, Ground truth labels as a numpy array of 0-s and 1-s.
        :return: dict
                    {'f1_score': 0.3, 'logloss': 0.7}
        """
        # Using decorator instead for data pre-processing
        # X = self.preprocess.transform(X)

        return {'f1_score': f1_score(y, self.predict(X)), 'logloss': log_loss(y, self.predict_proba(X))}

    # ------------------------------------------------------------------
    @robo_preprocess('X')
    def tune_parameters(self, X, y, hypeparam_grid=None):
        """
        Run K-fold cross validation to choose the best parameters

        Note: Output the average scores across all CV validation partitions and best parameters

        :param X: pd.DataFrame, Input features
        :param y: np.ndarray, Ground truth labels as a numpy array of 0-s and 1-s.
        :param hypeparam_grid: Dictionary of all hyper-parameter values for CV tuning.
        :return:  dict
                    {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag', ‘scores’:
                     {'f1_score': 0.3, 'logloss': 0.7}}
        """
        # Using decorator instead for data pre-processing
        # X = self.preprocess.transform(X)

        # Validate hypeparam_grid
        assert hypeparam_grid is None or not isinstance(hypeparam_grid, dict), \
            "hypeparam_grid should be either set to None to use default ranges or set as a dictionary of parameters"

        # If grid of hyper-parameters is not provided, use class default
        if hypeparam_grid is None:
            hypeparam_grid = self.hypeparam_grid

        scoring = {'f1_score': make_scorer(f1_score), 'logloss': make_scorer(log_loss)}

        # Define CV Grid Search
        clf_tuned = GridSearchCV(self, hypeparam_grid,
                                 scoring=scoring,
                                 cv=5,
                                 refit='logloss',
                                 iid=True,
                                 pre_dispatch='2*n_jobs',
                                 n_jobs=-1,
                                 verbose=True
                                 )
        # Execute CV
        clf_tuned.fit(X, y)

        # Get dictionary of best parameters
        tuned_dict = clf_tuned.best_params_

        # Update Original Estimator with best parameters (Optional)
        self.set_params(**clf_tuned.best_params_)
        self.fit(X, y)

        # Add average scores on all test partitions
        tuned_scores = {}
        for k in scoring.keys():
            tuned_scores[k] = clf_tuned.cv_results_['mean_test_%s' % k].mean()
        tuned_dict['scores'] = tuned_scores

        return tuned_dict


# ====================================================================================
# ====================================================================================
#                                 Main Part
# ====================================================================================
# ====================================================================================
# if __name__ == '__main__':
#     print("Executing  Model Example!")
#     from sklearn.datasets import load_breast_cancer  # Binary Class
#     test0 = 1
#
#     if test0:
#         df = pd.read_csv("https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club_reduced.csv", index_col=0,
#                          na_values=['na', 'nan', 'none', 'NONE'])
#         print("df Columns: ", len(df.columns.values))
#
#         col_y = 'is_bad'
#         y = np.ravel(df[col_y])
#         X = df.drop([col_y] + ['initial_list_status', 'pymnt_plan'], axis=1)
#
#         # X, y = load_breast_cancer(return_X_y=True)
#         X = pd.DataFrame(X)
#         # y = y[:-2]
#
#         # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
#         clf = RoboLogistic(solver='lbfgs',
#                            multi_class='auto',
#                            C=1.0,
#                            max_iter=500,
#                            scaler=StandardScaler())
#
#         clf.fit(X, y)
#         yhat = clf.predict(X)
#         p = clf.predict_proba(X)
#         score = clf.evaluate(X, y)
#         tune = clf.tune_parameters(X, y)
#         try:
#             print("\n X  %s: \n %s" % (X.shape, X[:5, :4]))
#         except:
#             pass
#
#         print("\n y  %s: \n %s" % (y.shape, y[:10]))
#         print("\n yhat %s: \n %s" % (yhat.shape, yhat[:10]))
#         print("\n p  %s: \n %s" % (p.shape, p[:10]))
#         print("\n score: %s" % score)
#         print("\n tune: %s" % tune)
#         print("\n Instance params: %s" % clf.get_params())
