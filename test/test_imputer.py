# -------------------------------------------------------------------------------
# Name:        RoboImputer Unit Tests
# Purpose:     Unit Test Cases to check RoboImputer Class
#
# Author:      Mohammad Maysami
#
# Created:     June 2019
# Copyright:   (c) MM 2019
# Licence:     See Git
# -------------------------------------------------------------------------------

import unittest
import numpy as np
import pandas as pd

# import os
# import sys
# targetPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(targetPath)
# sys.path.append(targetPath + "/..")
from src import RoboImputer
from sklearn.exceptions import NotFittedError
from test.data import TestData


# ======================================================================
class TestRunner(unittest.TestCase):

    # ------------------------------------------------------------------
    def setUp(self):
        self.numberThresh = .001
        self.places = 2  # Decimal Places

        # Define Sample Data
        a = ['a', 'b', 'b', np.nan]
        b = [1, 1, 2, np.nan]
        c = [1.1, 2.1, 2.1, np.nan]

        self.ma = 'b'
        self.mb = np.nanmean(b)
        self.mc = np.nanmean(c)
        self.X = pd.DataFrame({'a': a, 'b': b, 'c': c})


    # ------------------------------------------------------------------
    def test_bad_input_shape(self):
        # Check if transform method input shape mismatches fit method input
        # and it will raise ValueError("Shape of input ...")

        # Fit
        robo = RoboImputer()
        robo.fit(self.X)

        self.assertRaisesRegex(ValueError, "Shape of input", robo.transform, self.X.iloc[::2, :])
        self.assertRaisesRegex(ValueError, "Shape of input", robo.transform, self.X.iloc[:, ::2])

    # ------------------------------------------------------------------
    def test_unfit_instance(self):
        # Check if unfitted instance is called,
        # then it will raise NotFittedError!

        robo = RoboImputer()
        self.assertRaises(NotFittedError, robo.transform, self.X)

    # ------------------------------------------------------------------
    def test_fillvalue1(self):
        # Test a basic case

        # Fit
        Xt = RoboImputer().fit_transform(self.X)

        # assert Xt.loc[3, 'a'] == self.ma
        # assert abs(Xt.loc[3, 'b'] - self.mb) < self.numberThresh
        # assert abs(Xt.loc[3, 'c'] - self.mc) < self.numberThresh

        self.assertEqual(Xt.iloc[3, 0], self.ma)
        self.assertAlmostEqual(Xt.iloc[3, 1], self.mb, self.places)
        self.assertAlmostEqual(Xt.iloc[3, 2], self.mc, self.places)

    # ------------------------------------------------------------------
    def test_fillvalue2(self):
        # Test a more complex case

        # Load Test Data
        data = TestData(skcancer=False)
        df = data.df

        # Fit
        robo = RoboImputer()
        robo.fit(df)

        for c, col in enumerate(df.columns.values):
            if col in data.cols_numeric:
                self.assertAlmostEqual(robo.fillvalue[c], np.nanmean(df[col]), self.places)

        res_categorical = {
            'home_ownership': 'RENT',
            'verification_status': 'not verified',
            'pymnt_plan': 'n',
            'purpose_cat': 'debt consolidation',
            'zip_code': '100xx',
            'addr_state': 'CA',
            'initial_list_status': 'f',
            'policy_code': 'PC3',
        }

        if not data.skcancer:
            for col in res_categorical.keys():
                c = df.columns.get_loc(col)
                self.assertEqual(robo.fillvalue[c], res_categorical[col])
