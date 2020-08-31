import unittest

import numpy as np

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
    def ndim(self):
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

    def test_sample(self):
        mean, cov = np.zeros(TEST_NDIM), np.eye(TEST_NDIM)
        randvar = Normal(mean, cov)
        samp = self.mcm.sample(0.0, 1.0, 0.01, randvar.mean)
        self.assertEqual(samp.ndim, 1)
        self.assertEqual(samp.shape[0], TEST_NDIM)

        if VISUALISE is True:
            plt.title("100 Samples of a Mock Object")
            plt.plot(samp)
            plt.show()

    def test_ndim(self):
        self.assertEqual(self.mcm.ndim, TEST_NDIM)


class DeterministicModel(continuousmodel.ContinuousModel):
    """
    Deterministic Model: The (S)DE is
    dx(t) = x(t)dt + 0*dB(t),
    where B(t) is standard Brownian motion with diffusion
    equal to 1.

    Its solution and any sample---provided that the
    intiial distribution is a Dirac---should coincide
    with exponential function.
    """

    def drift(self, time, state, **kwargs):
        """
        Identity drift
        """
        return state

    def dispersion(self, time, state, **kwargs):
        """
        Identity dispersion
        """
        return 0 * np.eye(len(state), 1)

    @property
    def diffusionmatrix(self):
        """
        Unit diffusion
        """
        return np.eye(1)

    @property
    def ndim(self):
        """
        Unit diffusion
        """
        return TEST_NDIM


class TestDeterministicModel(unittest.TestCase):
    """
    Dirac initial distribution, l(t, x(t)) == 0
    and f(t, x(t)) = x(t) should yield exponential
    function as a sample.
    """

    def setUp(self):
        dm = DeterministicModel()
        randvar = Dirac(np.ones(TEST_NDIM))
        self.samp = dm.sample(0.0, 1.0, 0.01, randvar.mean)

    def test_sample_shape(self):
        self.assertEqual(self.samp.ndim, 1)
        self.assertEqual(self.samp.shape[0], TEST_NDIM)

    def test_sample_vals(self):
        diff = np.abs(np.exp(1) - self.samp[0])
        self.assertLess(diff, 1e-1)
