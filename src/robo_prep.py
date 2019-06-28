import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


# ======================================================================
class DataFrameImputer(TransformerMixin):
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

