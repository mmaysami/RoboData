# -------------------------------------------------------------------------------
# Name:        RoboLogistic Unit Tests
# Purpose:     Unit Test Cases to check RoboLogistic Class
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
import pandas as pd

# import os
# import sys
# targetPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(targetPath)
# sys.path.append(targetPath + "/..")
from src import RoboLogistic
from sklearn.exceptions import NotFittedError
from test.data import TestData


# ======================================================================
class TestRunner(unittest.TestCase):

    # ------------------------------------------------------------------
    def setUp(self):
        self.numberThresh = .001
        self.places = 2  # Decimal Places

    # ------------------------------------------------------------------
    def test_bad_input_shape(self):
        # Check if transform method input shape mismatches fit method input
        # and it will raise ValueError("Shape of input ...")

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        # Fit
        robo = RoboLogistic()
        robo.fit(X, y)

        # Only Columns, shape[1] should be the same
        self.assertRaises(ValueError, robo.predict, X.iloc[:, ::2])
        self.assertRaises(ValueError, robo.predict_proba, X.iloc[:, ::2])

    # ------------------------------------------------------------------
    def test_unfit_instance(self):
        # Check if unfitted instance is called,
        # then it will raise NotFittedError!

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic()
        self.assertRaises(NotFittedError, robo.predict, X)
        self.assertRaises(NotFittedError, robo.predict_proba, X)
        self.assertRaises(NotFittedError, robo.evaluate, X, y)

    # ------------------------------------------------------------------
    def test_result_format_fit(self):

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic(random_state=0)

        # Fit returns None
        res_fit = robo.fit(X, y)
        self.assertIsNone(res_fit)

    # ------------------------------------------------------------------
    def test_result_format_predict(self):

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic(random_state=0)
        robo.fit(X, y)

        # Output Type np.ndarray
        # Output Example np.array([1, 0, 1])
        res_predict = robo.predict(X)
        self.assertIsInstance(res_predict, np.ndarray)
        self.assertTupleEqual(res_predict.shape, (data.df.shape[0], ))
        # self.assertFalse(np.any(res_predict != 0 & res_predict != 0))

    # ------------------------------------------------------------------
    def test_result_format_predict_proba(self):

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic(random_state=0)
        robo.fit(X, y)
        # Output Type np.ndarray
        # Output Example np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        res_predict_proba = robo.predict_proba(X)
        self.assertIsInstance(res_predict_proba, np.ndarray)
        self.assertEqual(res_predict_proba.shape, (data.df.shape[0], len(robo.classes_)))
        self.assertTupleEqual(res_predict_proba.shape, (data.df.shape[0], len(robo.classes_)) )
        # self.assertFalse(np.any(np.sum(res_predict_proba, axis=1) != 1))

    # ------------------------------------------------------------------
    def test_result_format_evaluate(self):

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic(random_state=0)
        robo.fit(X, y)

        # Output Type dict
        # Output Example {'f1_score': 0.3, 'logloss': 0.7}
        res_evaluate = robo.evaluate(X, y)
        self.assertIsInstance(res_evaluate, dict)
        self.assertTrue('f1_score' in res_evaluate.keys())
        self.assertTrue('logloss' in res_evaluate.keys())
        self.assertIsInstance(res_evaluate['f1_score'], np.float)
        self.assertIsInstance(res_evaluate['logloss'], np.float)

    # ------------------------------------------------------------------
    def test_result_format_tune(self):

        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic(random_state=0)

        # Output Type dict
        # Output Example {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag', ‘scores’:
        # {'f1_score': 0.3, 'logloss': 0.7}}
        res_tune_parameters = robo.tune_parameters(X, y)
        self.assertIsInstance(res_tune_parameters, dict)
        self.assertTrue('scores' in res_tune_parameters.keys())
        self.assertTrue('f1_score' in res_tune_parameters['scores'].keys())
        self.assertTrue('logloss' in res_tune_parameters['scores'].keys())

    # ------------------------------------------------------------------
    def test_reproduce(self):
        # Load Test Data
        data = TestData()
        X, y = data.make_X_y()

        robo = RoboLogistic(random_state=10)
        robo.fit(X, y)
        res_evaluate = robo.evaluate(X, y)

        if not data.skcancer:
            self.assertDictEqual(res_evaluate, {'f1_score': 0.003084040092521203, 'logloss': 0.37192273211820986})
        else:
            self.assertDictEqual(res_evaluate, {'f1_score': 0.9902370990237099, 'logloss': 0.05331853517496121})

