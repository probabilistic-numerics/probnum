import unittest

import numpy as np
import probnum.random_variables as rvs

from probnum.random_variables import Normal, Dirac
from probnum.filtsmooth.statespace.continuous import continuousmodel

VISUALISE = False
TEST_NDIM = 10

if VISUALISE is True:
    import matplotlib.pyplot as plt


class MockContinuousModel(continuousmodel.ContinuousModel):
    """
    Minimal implementation required to make a cont.
    model work.

    SDE is
    dx(t) = x(t)dt + x(t)dB(t),
    where B(t) is standard 1d-Brownian motion with diffusion
    equal to 1.
    """

    def transition_rv(self, rv, **kwargs):
        return rv

    def transition_realization(self, real, **kwargs):
        return rvs.asrandvar(real)


    def drift(self, time, state, **kwargs):
        """
        Identity drift
        """
        return state

    def dispersion(self, time, state, **kwargs):
        """
        Identity dispersion
        """
        return np.eye(len(state), 1)

    @property
    def diffusionmatrix(self):
        """
        Unit diffusion
        """
        return np.eye(1)

    @property
    def dimension(self):
        """
        Unit diffusion
        """
        return TEST_NDIM


class TestContinuousModel(unittest.TestCase):
    """
    Test essential functionalities of continuous model.
    """

    def setUp(self):
        self.mcm = MockContinuousModel()


    def test_call_rv(self):
        out = self.mcm(rvs.Dirac(0.1))
        self.assertIsInstance(out, rvs.RandomVariable)

    def test_call_arr(self):
        out = self.mcm(np.random.rand(4))
        self.assertIsInstance(out, rvs.RandomVariable)

    def test_dimension(self):
        self.assertEqual(self.mcm.dimension, TEST_NDIM)
