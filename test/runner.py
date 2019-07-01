
# import unittest
from unittest import TestLoader, TestSuite, TextTestRunner


# import test modules
from test import test_imputer
from test import test_featurizer
from test import test_logistic

# initialize the test suite
loader = TestLoader()
suite = TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_imputer))
suite.addTests(loader.loadTestsFromModule(test_featurizer))
suite.addTests(loader.loadTestsFromModule(test_logistic))
# suite.addTests(TestLoader().discover('.'))

# initialize a runner, pass it your suite and run it
# print("Testing in progress:")
runner = TextTestRunner(verbosity=3)
result = runner.run(suite)
# unittest.main(testRunner=runner)
