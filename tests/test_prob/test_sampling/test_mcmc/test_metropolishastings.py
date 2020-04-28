"""
We only do essential testing here, as in:
Does the algorithm complain about wrong inputs?
"""

import unittest

import numpy as np

from probnum.optim import objective
from probnum.prob.sampling.mcmc import metropolishastings


class MockMetropolisHastings(metropolishastings.MetropolisHastings):
    """
    Mock object for basic unittesting.
    """

    def generate_proposal(self, currstate, pwidth, *pars, **namedpars):
        """
        Does nothing really.
        We don't need it to do anything but
        it has to be here for "@abstractmethod" reasons.
        """
        return currstate, 0.0


class TestMetropolisHastings(unittest.TestCase):
    """
    Tests whether sample_nd(...) complains if 
        * logdens does not evalaute to a scalar
        * the initial state is no ndarray of shape (d,)
    """

    def test_logdens_no_autodiff(self):
        """
        """

        def no_ad(x):
            return x.T @ x

        initstate = np.ones(1)
        with self.assertRaises(ValueError):
            methast = MockMetropolisHastings(no_ad)

        ad = objective.Objective(no_ad)  # control variate
        methast = MockMetropolisHastings(ad)

    def test_initstate_no_array(self):
        """
        """

        def obj(x):
            return x.T @ x

        logdens = objective.Objective(obj)
        methast = MockMetropolisHastings(logdens)
        good_init = np.ones(1)
        bad_init = 1.0
        with self.assertRaises(ValueError):
            methast.sample_nd(100, bad_init, 0.1)
        methast.sample_nd(100, good_init, 0.1)  # control variate

    def test_evaluate_no_scalar(self):
        """
        """

        def no_ad(x):
            return np.array(x)

        initstate = np.ones(1)
        ad = objective.Objective(no_ad)
        methast = MockMetropolisHastings(ad)
        with self.assertRaises(TypeError):
            methast.sample_nd(ad, 100, initstate, 0.1)
