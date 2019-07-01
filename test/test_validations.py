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
import unittest
import numpy as np
import pandas as pd

targetPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(targetPath)
sys.path.append(targetPath + "/..")
from src import RoboImputer, RoboFeaturizer, RoboLogistic
from test.data import TestData

# ======================================================================
class TestRunner(unittest.TestCase):
    def setUp(self):
        self.numberThresh = .001
        self.places = 3  # Decimal Places

        self.data = TestData()

    # ------------------------------------------------------------------
    def test_imputer1(self):
        a = ['a', 'b', 'b', np.nan]
        b = [1, 1, 2, np.nan]
        c = [1, 2, 2, np.nan]
        ma = 'b'
        mb = np.nanmean(b)
        mc = np.nanmean(c)
        X = pd.DataFrame({'a': a, 'b': b, 'c': c})
        Xt = RoboImputer().fit_transform(X)
        assert Xt.loc[3, 'a'] == ma
        assert abs(Xt.loc[3, 'b'] - mb) < self.numberThresh
        assert abs(Xt.loc[3, 'c'] - mc) < self.numberThresh
        self.assertEqual(Xt.iloc[3, 0], 'b')
        self.assertAlmostEqual(Xt.iloc[3, 1], 1.333, self.places)
        self.assertAlmostEqual(Xt.iloc[3, 2], 1.667, self.places)

    # ------------------------------------------------------------------
    def test_featurizer1(self):

        Xt = RoboImputer().fit_transform(self.data.df)
        pass
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def sumAbsError(self, l1, l2):
        l = [abs(x - y) for x, y in zip(l1, l2)]
        return sum(l)

    # ------------------------------------------------------------------
    @staticmethod
    def isListEqual(self, list1, list2):
        l = [x - y for x, y in zip(list1, list2)]
        return sum(l) == 0

    # ------------------------------------------------------------------
    def isEqual_myRcursive(self, d1, d2):
        if type(d1) == type(d2) == type(dict()):
            for key in d1:
                if not self.isEqual_myRcursive(d1[str(key)], d2[str(key)]):
                    return False
            return True
        elif type(d1) == type(d2) == type(list()):
            for i in range(len(d1)):
                if not self.isEqual_myRcursive(d1[i], d2[i]):
                    return False
            return True

        elif type(d2) in [type(1), type(1.1), type(np.float64())] and type(d1) in [type(1), type(1.1), type(np.float64())]:
            return abs(d1 - d2) < self.numberThresh
        elif type(d1) == type(d2) == type('1'):
            return d1 == d2
        else:
            return False


# ======================================================================
#               Main
# ======================================================================
if __name__ == '__main__':
    print("Testing Validation Suit:")
    runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
