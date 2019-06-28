import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.linear_model import LogisticRegression
import scipy

# ======================================================================
class RoboImputer(TransformerMixin):
    """
        Simple Class of Missing Value Imputing
        Column dtype    :   Imputing Value
        object/category :   most frequent value in the column
        float           :   mean of column values
        other dtype     :   median of column
    """

    def __init__(self):
        self.fillvalue = None

    # ------------------------------------------------------------------
    @staticmethod
    def most_frequent(col):
        try:
            return col.value_counts().index[0]
        except IndexError:
            return 0

    # ------------------------------------------------------------------
    def get_impute_fill_value(self, col):
        if col.dtype == np.dtype("O"):
            val = self.most_frequent(col)
        elif str(col.dtype) == "category":
            val = self.most_frequent(col)
        elif col.dtype == np.dtype(float):
            val = col.mean()
        else:
            val = col.median()
        if isinstance(val, float) and np.isnan(val):
            val = 0
        return val

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.fillvalue = pd.Series([self.get_impute_fill_value(X[c]) for c in X], index=X.columns)
        return self

    # ------------------------------------------------------------------
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.fillna(self.fillvalue, inplace=False)

# ======================================================================
class RoboFeaturizer(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self,
                 max_unique_for_discrete=10,
                 max_missing_to_keep=0.80,
                 add_missing_flag=False,
                 encode_categorical=False,
                 max_category_for_ohe=10,
                 sparse=False):

        self.max_unique_for_discrete = max_unique_for_discrete
        self.max_missing_to_keep = max_missing_to_keep
        self.add_missing_flag = add_missing_flag
        self.encode_categorical = encode_categorical
        self.max_category_for_ohe = max_category_for_ohe
        self.sparse = sparse
        self.imputer = None

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
        # --------------------------------
        def is_numeric(col):
            def try_numeric(x):
                try:
                    float(x)
                    return True
                except ValueError:
                    return False

            return np.all([try_numeric(x) for x in col])

        # --------------------------------
        def is_integer(col):
            def try_int_comparison(x):
                try:
                    return int(x) == float(x)
                except ValueError:
                    return False

            return np.all([try_int_comparison(x) for x in col])

            # --------------------------------
        # Ensure X is DataFrame and Get Columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        columns = list(X.columns.values)

        # Init Column Related Attribute
        self.drop_cols = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Columns to drop (1 Unique)
        self.ohe_indices = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Columns needing One-Hot Encoding
        self.lbl_indices = np.zeros(X.shape[1], dtype=bool)  # Bool Index of Columns needing Label Encoding
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
        self.imputer = RoboImputer()
        X = self.imputer.fit_transform(X)

        # Process Columns: Drop Uni-Valued Variables, Detect Cont. vs Boolean Dummy
        for i, col in enumerate(X):
            # dropped due to high missing
            if self.drop_cols[i]:
                pass

            # Single Value Columns
            elif len(set(X[col])) == 1:
                self.drop_cols[i] = True

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
        return self
