# -------------------------------------------------------------------------------
# Name:        RoboFeaturizer Unit Tests
# Purpose:     Unit Test Cases to check RoboFeaturizer Class
#
#
# Author:      Mohammad Maysami
#
# Created:     June 2019
# Copyright:   (c) MM 2019
# Licence:     See Git
# -------------------------------------------------------------------------------

import unittest
import numpy as np
# import pandas as pd

# import os
# import sys
# targetPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(targetPath)
# sys.path.append(targetPath + "/..")
from src import RoboFeaturizer
from sklearn.exceptions import NotFittedError
from test.data import TestData


# ======================================================================
class TestRunner(unittest.TestCase):

    # ------------------------------------------------------------------
    def setUp(self):
        self.numberThresh = .001
        self.places = 3  # Decimal Places

    # ------------------------------------------------------------------
    def test_bad_input_shape(self):
        # Check if transform method input shape mismatches fit method input
        # and it will raise ValueError("Shape of input ...")

        # Load Test Data
        data = TestData()
        df = data.df

        # Fit
        robo = RoboFeaturizer()
        robo.fit(df)

        self.assertRaisesRegex(ValueError, "Shape of input", robo.transform, df.iloc[::2, :])
        self.assertRaisesRegex(ValueError, "Shape of input", robo.transform, df.iloc[:, ::2])

    # ------------------------------------------------------------------
    def test_unfit_instance(self):
        # Check if unfitted instance is called,
        # then it will raise NotFittedError!

        # Load Test Data
        data = TestData()
        df = data.df

        robo = RoboFeaturizer()
        self.assertRaises(NotFittedError, robo.transform, df)

    # ------------------------------------------------------------------
    def test_missing_value(self):
        # Test if class handles new category level at prediction time

        # Load Test Data
        data = TestData(skcancer=False)
        df = data.df

        if data.skcancer:
            return True

        cols_pick = ['policy_code', 'addr_state', 'annual_inc']
        X = df.loc[:, cols_pick]

        # Fit
        robo = RoboFeaturizer(encode_categorical=False)
        robo.fit(X)

        # Create prediction X with missing and new labels
        Xp = X.copy()
        # range0 = range(1, 10)
        # range1 = range(10, 20)
        range2 = range(20, 30)

        # Xp.loc[range0, cols_pick[0]] = "PCNEW TESTVALUE"
        # Xp.loc[range1, cols_pick[1]] = "MNPQ TESTVALUE"
        Xp.loc[range2, cols_pick[2]] = np.NaN
        Xpt = robo.transform(Xp)

        for row in range2[1:-1]:
            self.assertIsInstance(Xpt[row, 0], np.float)
            self.assertAlmostEqual(robo.scaler.inverse_transform([Xpt[row, 0]])[0], robo.imputer.fillvalue[2], self.places)

    # ------------------------------------------------------------------
    def test_new_category(self):
        # Test if class handles new category level at prediction time

        # Load Test Data
        data = TestData(skcancer=False)
        df = data.df

        if data.skcancer:
            return True

        cols_pick = ['policy_code']  # , 'addr_state', 'annual_inc']
        X = df.loc[:, cols_pick]

        # Fit
        robo = RoboFeaturizer(encode_categorical=True)
        robo.fit(X)

        # Create prediction X with missing and new labels
        Xp = X.copy()
        range0 = range(1, 10)
        Xp.loc[range0, cols_pick[0]] = "PCNEW TESTVALUE"
        Xpt = robo.transform(Xp)

        for row in range0:
            self.assertIsInstance(Xpt[row, 0], np.float)
