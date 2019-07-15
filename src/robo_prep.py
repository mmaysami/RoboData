# -------------------------------------------------------------------------------
# Name:        Robo Data Prep
# Purpose:     Imputing and Featurizing Input DataFrame (for RoboLogistics)
#
#
# Author:      Mohammad Maysami
#
# Created:     June 2019
# Copyright:   (c) MM 2019
# Licence:     See Git
# -------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy as scp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted  # , check_consistent_length, check_X_y, check_array
from sklearn.exceptions import NotFittedError

# Turn off warning as value settings inside classes follow proper pandas format
# But it will warn if input of functions are copy themselves.
pd.options.mode.chained_assignment = None


# ======================================================================
class RoboImputer(TransformerMixin):
    """
        Simple Class of Missing Value Imputing
        Column dtype    :   Imputing Value
        object/category :   most frequent value in the column
        float           :   mean of column values
        other dtype     :   median of column

        :param max_unique_for_discrete: [int], Encode columns of up to this threshold of unique numeric values, using one-hot-encoding.
    """

    def __init__(self, max_unique_for_discrete=10):
        # attribute defined in fit to raise error otherwise
        # self.fillvalue = None

        self.max_unique_for_discrete = max_unique_for_discrete
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def most_frequent(col):
        """
        Find the most frequest value in a pandas DataFrame Columns

        :param col: Pandas DataFrame Column
        :return: Most frequent value in input column
        """
        try:
            return col.value_counts().index[0]
        except IndexError:
            return 0

    # ------------------------------------------------------------------
    def get_impute_fill_value(self, col):
        """
        Get fill value for any missing sample in the given column of data

        :param col: Columns of data as pandas DataSeries
        :return: Value to be used to fill missing samples
        """
        if col.dtype == np.dtype("O"):
            val = self.most_frequent(col)
        elif str(col.dtype) == "category":
            val = self.most_frequent(col)
        # TODO: Improve Comparison
        elif col.dtype in [np.dtype(np.float),
                           np.dtype(np.float16),
                           np.dtype(np.float32),
                           np.dtype(np.float64)]:
            val = col.mean()
        elif col.dtype in [np.dtype(np.int),
                           np.dtype(np.int8),
                           np.dtype(np.int16),
                           np.dtype(np.int32),
                           np.dtype(np.int64)] and \
                col.nunique() > self.max_unique_for_discrete:
            val = col.mean()
        else:
            val = col.median()
        if isinstance(val, float) and np.isnan(val):
            val = 0
        return val

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit/Find fill values for all columns of training input data

        :param X: pd.DataFrame, Input features
        :param y: None, not used in this implementation
        :return: None, Update/Fit instance of model class
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Set shape of fitting input to check versus tranform input
        self.fit_X_shape_ = X.shape

        self.fillvalue = pd.Series([self.get_impute_fill_value(X[c]) for c in X], index=X.columns)
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Check is fit had been called
        check_is_fitted(self, ['fillvalue'])

        # Check that the input is of the same shape as the one passed during fit.
        if X.shape != self.fit_X_shape_:
            raise ValueError('Shape of input %s is different from what was seen in `fit` %s' % (X.shape, self.fit_X_shape_))

        return X.fillna(self.fillvalue, inplace=False)


# ======================================================================
class RoboFeaturizer(BaseEstimator, TransformerMixin):
    """
        Data Cleaning Class
        Fill-in missing values
        Add binary column to identify missing values
        Convert nominal numbers using One-Hot-Encoder
        Convert categorical columns using either Label or One-Hot Encdoers
        Drop columns with no variation or  major missing values


        :param max_unique_for_discrete: [int], Encode columns of up to this threshold of unique numeric values, using one-hot-encoding.
        :param max_missing_to_keep: [0=<float<=1], Drop columns if fraction (%) of missing values are higher than this threshold
        :param add_missing_flag: [Bool], For columns with missing values smaller than the threshold, add a binary column to mark missing data points.
        :param encode_categorical: [Bool], Encode categorical (non-numerical) columns
        :param max_category_for_ohe: [int], Encode categorical columns with number of categories up to this threshold,
                    use one-hot-encoding and for the rest use alternate. Only effective if `encode_categorical=True`
        :param scaler: [None or sklearn scaler],  No scaling of input if None, scale input using 'fit_transform' method
    """

    def __init__(self,
                 max_unique_for_discrete=10,
                 max_missing_to_keep=0.80,
                 add_missing_flag=False,
                 encode_categorical=True,
                 max_category_for_ohe=10,
                 scaler=StandardScaler()
                 ):

        # Validate Initialization Parameters (Only additional ones to LogisticRegression)
        assert max_unique_for_discrete >= 0, "max_unique_for_discrete must be non-negative integer."
        assert 0 <= max_missing_to_keep <= 1, "max_missing_to_keep must be between 0 and 1."
        assert isinstance(add_missing_flag, bool), "add_missing_flag must be boolean."
        assert isinstance(encode_categorical, bool), "encode_categorical must be boolean."
        assert max_category_for_ohe >= 0, "max_category_for_ohe must be none-negative integer."
        # TODO: Test and add more scalers or make assertion more flexible to accept more choices
        assert (scaler is None) or isinstance(scaler, (Normalizer, StandardScaler, MinMaxScaler)), \
            "scaler must be either None or one of predefined sklearn Scalers"

        self.max_unique_for_discrete = max_unique_for_discrete
        self.max_missing_to_keep = 1  # max_missing_to_keep
        self.add_missing_flag = add_missing_flag
        self.encode_categorical = encode_categorical
        self.max_category_for_ohe = max_category_for_ohe
        self.scaler = scaler
        self.imputer = RoboImputer(max_unique_for_discrete=max_unique_for_discrete)

        self.one_hot_encoder = None
        self.label_encoder = None
        self.feature_indices_ = None
        self.feature_names_ = None
        self.missing_col_names = None

        self.drop_cols = None
        self.ohe_indices = None
        self.lbl_indices = None
        self.numeric_indices = None

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit/pre-analyze training input data

        :param X: pd.DataFrame, Input features
        :param y: None, not used in this implementation
        :return: None, Update/Fit instance of model class
        """

        # TODO: Move Util functions into a separate file for re-use
        # --------------------------------
        def is_numeric(col):
            def is_floatable(x):
                try:
                    float(x)
                    return True
                except ValueError:
                    return False

            return np.all([is_floatable(e) for e in col])

        # --------------------------------
        def is_integer(col):
            def is_int_eq_float(x):
                try:
                    return int(x) == float(x)
                except ValueError:
                    return False

            return np.all([is_int_eq_float(x) for x in col])

        # --------------------------------
        # Ensure X is DataFrame and Get Columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Set Transformer Template Style
        # Input validation (Not Applicable for Categorical DataFrame)
        # X = check_array(X)

        # Not using X_ since we care about consistent shape of X_fit and X_transform
        # check_consistent_length on X only checks length
        # self.X_ = X
        self.fit_X_shape_ = X.shape

        # Init Column Related Attribute
        columns = list(X.columns.values)
        self.drop_cols = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Columns to drop (1 Unique)
        self.ohe_indices = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Columns needing One-Hot Encoding
        self.lbl_indices = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Columns needing Label Encoding
        # TODO: Replace Lable with Target or Better Encoders
        self.numeric_indices = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Numeric Columns
        self.categorical_indices = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Categorical Columns

        self.feature_indices_ = []  # Index of Selected Feature (Not Dropped)
        self.feature_names_ = []  # Modified Name of Features with Missing, Continuous, Dummy ...
        self.missing_col_names = []  # Name of Columns with Missing Data

        # Get Name of Columns with Missing Value
        for col_name in X.columns[np.any(pd.isnull(X), axis=0)]:
            self.missing_col_names.append(col_name)

            if X[col_name].isna().sum() > self.max_missing_to_keep * X.shape[0]:
                i = X.columns.get_loc(col_name)
                self.drop_cols[i] = True

            elif self.add_missing_flag:
                self.feature_names_.append(str(col_name) + "_missing")
                self.feature_indices_.append(columns.index(col_name))

        # Impute Missing data
        X = self.imputer.fit_transform(X)
        if X.isnull().values.any():
            raise (NotFittedError("Input X has null values. Check if RoboImputer instance is fitted."))

        # Process Columns: Drop Uni-Valued Variables, Detect Cont. vs Boolean Dummy
        for i, col in enumerate(X):
            # dropped due to high missing
            if self.drop_cols[i]:
                pass

            # TODO: This might cause issues in CV
            # # Single Value Columns
            # elif len(set(X[col])) == 1:
            #     self.drop_cols[i] = True

            # Numeric Columns
            elif is_numeric(X[col]):
                if len(set(X[col])) > 2:
                    self.feature_names_.append("{}_{}_continuous".format(col, i))
                else:
                    self.feature_names_.append("{}_{}_dummy".format(col, i))
                self.feature_indices_.append(i)

                num_unique = len(np.unique(X[col]))
                if is_integer(X[col]) and 3 <= num_unique <= self.max_unique_for_discrete:
                    self.ohe_indices[i] = True
                self.numeric_indices[i] = True

            elif self.encode_categorical:
                self.categorical_indices[i] = True

                num_unique = len(np.unique(X[col]))
                if num_unique <= self.max_category_for_ohe:
                    self.ohe_indices[i] = True
                else:
                    self.lbl_indices[i] = True

        # One-Hot-Encoding
        if np.any(self.ohe_indices):
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
            self.one_hot_encoder.fit(X[X.columns[self.ohe_indices]])

            for cat, att, ind in zip(self.one_hot_encoder.categories_,
                                     X.columns[self.ohe_indices],
                                     np.where(self.ohe_indices)[0]):
                self.feature_names_.extend(["{}_OHE_{}".format(att, j) for j in cat])
                self.feature_indices_.extend([ind] * len(cat))

        # Label Encoding
        if np.any(self.lbl_indices):
            # Dictionary of Label Encoders for Each Column
            self.label_encoder = dict()

            for ind in np.where(self.lbl_indices)[0]:
                col_lbl_encoder = LabelEncoder()
                col_lbl_encoder.fit(X[X.columns[ind]])
                self.feature_names_.extend(["{}_LBL_{}".format(X.columns[ind], len(col_lbl_encoder.classes_))])
                self.feature_indices_.extend([ind])
                self.label_encoder[ind] = col_lbl_encoder

        # Update Name / Index of Features
        self.feature_names_ = np.array(self.feature_names_)
        self.feature_indices_ = np.array(self.feature_indices_)

        # Scaling should be done in transform so that all columns are converted to numerical format!
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        """
        Preprocess and transform raw input X to final numeric and filled form

        :param X: pd.DataFrame, Input features
        :return: pd.DataFrame, Processed and Featurized Input features
        """
        # Check is fit had been called
        # Not using X_ since we care about consistent shape of X_fit and X_transform
        # check_consistent_length on X only checks length
        #  check_is_fitted(self, ['X_'])
        check_is_fitted(self, ['fit_X_shape_'])

        # Input validation (Not Applicable for Categorical DataFrame)
        # X = check_array(X)

        # Check that the input is of the same shape as the one passed during fit.
        # Not using X_ since we care about consistent shape of X_fit and X_transform
        # check_consistent_length on X only checks length
        # check_consistent_length(self.X_,X)
        if X.shape != self.fit_X_shape_:
            raise ValueError('Shape of input %s is different from what was seen in `fit` %s' % (X.shape, self.fit_X_shape_))

        # Ensure X is DataFrame and Get Columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        xt_numeric = []
        # Missing Columns
        if self.add_missing_flag:
            for col in self.missing_col_names:
                xt_numeric.append(pd.isnull(X[col]))

        # Drop New Categories for Label-Encoded Columns
        if np.any(self.lbl_indices):
            for ind in np.where(self.lbl_indices)[0]:
                cat = self.label_encoder[ind].classes_
                ind_new_cat = np.logical_not(X[X.columns[ind]].isin(cat)), X.columns[ind]
                X.loc[ind_new_cat] = np.NaN

        # Impute Missing Values
        X = self.imputer.transform(X)

        # Take Care of To-Be-Dropped and Numeric Columns
        for i, col in enumerate(X):
            # Drop Column
            if self.drop_cols[i]:
                continue

            # Append Numeric Columns
            if self.numeric_indices[i]:
                xt_numeric.append(X[col].astype(float))

        # Combine all columns of Xt
        xt_cols = []
        if xt_numeric:
            xt_cols.append(scp.sparse.coo_matrix(xt_numeric).T)

        # One-Hot-Encoding
        if np.any(self.ohe_indices):
            ohet = self.one_hot_encoder.transform(X[X.columns[self.ohe_indices]])
            xt_cols += [ohet]

        # Label-Encoding
        if np.any(self.lbl_indices):
            for ind in np.where(self.lbl_indices)[0]:
                # lblt (N,), needs to be converted to 2D
                lblt = self.label_encoder[ind].transform(X[X.columns[ind]])
                xt_cols += [lblt.reshape(-1, 1)]

        xt = scp.sparse.hstack(xt_cols).tocsr()
        # Convert data from sparse to regular array
        xt = xt.toarray()

        # Scaling is done only in transform where columns are all numeric
        if self.scaler is not None:
            xt = self.scaler.fit_transform(xt)

        return xt


# ====================================================================================
# ====================================================================================
#                                 Main Part
# ====================================================================================
# ====================================================================================
if __name__ == '__main__':
    # print("Quick Testing!")
    pass
