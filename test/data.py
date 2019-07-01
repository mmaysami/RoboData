# -------------------------------------------------------------------------------
# Name:        Robo Unit Tests
# Purpose:     Unit Test Cases to check whether RoboLogistics
#                 - is reproducible
#                 - can handle missing values
#                 - can handle new category levels at prediction time
#                 - returns results in the expected format
#                 - other useful unit tests you may think of (if time allows)
#
# Author:      Mohammad Maysami
#
# Created:     June 2019
# Copyright:   (c) MM 2019
# Licence:     See Git
# -------------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  # Binary Class


# ======================================================================
class TestData(object):
    def __init__(self, skcancer=False, verbose=False):

        self.skcancer = skcancer
        # Get data from source
        if not skcancer:
            try:
                self.df = pd.read_csv("https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club_reduced.csv", index_col=0,
                                      na_values=['na', 'nan', 'none', 'NONE'])
            except:
                self.df = pd.read_csv('../data/DR_Demo_Lending_Club_reduced.csv', index_col=0, na_values=['na', 'nan', 'none', 'NONE'])

        else:
            X, y = load_breast_cancer(return_X_y=True)
            self.df = pd.DataFrame(X)

        cols = [
            ('Id', 'Id', 'Numeric'),
            ('addr_state', 'Customer', 'Categorical'),
            ('annual_inc', 'Customer', 'Numeric'),
            ('collections_12_mths_ex_med', 'Customer', 'Numeric'),
            ('delinq_2yrs', 'Customer', 'Numeric'),
            ('emp_length', 'Customer', 'Numeric'),
            ('home_ownership', 'Customer', 'Categorical'),
            ('inq_last_6mths', 'Customer', 'Numeric'),
            ('mths_since_last_delinq', 'Customer', 'Numeric'),
            ('mths_since_last_major_derog', 'Customer', 'Numeric'),
            ('mths_since_last_record', 'Customer', 'Numeric'),
            ('open_acc', 'Customer', 'Numeric'),
            ('pub_rec', 'Customer', 'Numeric'),
            ('pymnt_plan', 'Customer', 'Categorical'),
            ('revol_bal', 'Customer', 'Numeric'),
            ('revol_util', 'Customer', 'Numeric'),
            ('total_acc', 'Customer', 'Numeric'),
            ('zip_code', 'Customer', 'Categorical'),
            ('debt_to_income', 'Loan', 'Numeric'),
            ('initial_list_status', 'Loan', 'Categorical'),
            ('policy_code', 'Loan', 'Categorical'),
            ('purpose_cat', 'Loan', 'Categorical'),
            ('verification_status', 'Loan', 'Categorical'),
            ('is_bad', 'Target', 'Numeric')]

        cols = pd.DataFrame(data=cols, columns=["Name", "Category", "Type"])

        # Create list of cols for subgroups
        self.cols = cols
        self.cols_customer = cols[cols["Category"] == "Customer"]["Name"].values
        self.cols_loan = cols[cols["Category"] == "Loan"]["Name"].values
        self.cols_categorical = cols[cols["Type"] == "Categorical"]["Name"].values
        self.cols_numeric = cols[cols["Type"] == "Numeric"]["Name"].values

        self.cols_y = ["is_bad"]
        self.cols_id = ['Id']
        self.cols_numeric = [e for e in self.cols_numeric if e not in self.cols_id + self.cols_y]

        # Print out list of columns for each subgroup
        if verbose:
            print("\n *** Customer: \n", self.cols_customer)
            print("\n *** Loan: \n", self.cols_loan)
            print("\n *** Categorical: \n", self.cols_categorical, "\n")
            print("\n *** Numerical: \n", self.cols_numeric)
            print("Number of Numeric Cols: ", len(self.cols_numeric))
        pass

    # ------------------------------------------------------------------
    def make_X_y(self):

        if not self.skcancer:
            self.y = np.ravel(self.df[self.cols_y])
            self.X = self.df.drop(self.cols_y, axis=1)
            # TODO: Removing Columns with Single Value causes error in CV, chec in future
            # self.X = self.df.drop(self.cols_y+['initial_list_status', 'pymnt_plan'], axis=1)

        else:
            self.X, self.y = load_breast_cancer(return_X_y=True)
            self.X = pd.DataFrame(self.X)

        return self.X, self.y

