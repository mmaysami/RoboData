
# import unittest
from unittest import TestLoader, TestSuite, TextTestRunner


# import test modules
from test import test_validations
from test import tests1

# initialize the test suite
loader = TestLoader()
suite  = TestSuite()

# add tests to the test suite
# suite.addTests(loader.loadTestsFromModule(test_validations))
# suite.addTests(loader.loadTestsFromModule(tests1))
suite.addTests(TestLoader().discover('.'))

# initialize a runner, pass it your suite and run it
print("Testing in progress:")
runner = TextTestRunner(verbosity=3)
result = runner.run(suite)
# unittest.main(testRunner=runner)
