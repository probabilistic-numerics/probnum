"""
"""

import unittest

import numpy as np

from probnum.filtsmooth import bayesianfilter
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal


class MockBayesianFilter(bayesianfilter.BayesianFilter):
    """
    Mock BayesianFilter object for testing.
    As little implementation as necessary.
    """

    def predict(self, start, stop, randvar, **kwargs):
        """
        """
        return randvar

    def update(self, time, randvar, data, **kwargs):
        """
        """
        return randvar


class TestBayesianFilter(unittest.TestCase):
    """
    Test if the mock object works.
    Somewhat meaningless, but it might be good to
    have a first unit test in place.
    """

    def setUp(self):
        """
        """
        self.mbf = MockBayesianFilter()

    def test_predict(self):
        """
        """
        mean, cov = np.zeros(2), np.eye(2)
        mvg = RandomVariable(distribution=Normal(mean, cov))
        self.mbf.predict(start=0.1, stop=0.2, randvar=mvg)

    def test_update(self):
        """
        """
        mean, cov = np.zeros(2), np.eye(2)
        mvg = RandomVariable(distribution=Normal(mean, cov))
        self.mbf.update(time=0.1, randvar=mvg, data=None)


if __name__ == '__main__':
    unittest.main()
