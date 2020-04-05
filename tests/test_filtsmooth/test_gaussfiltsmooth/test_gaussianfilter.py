"""
Rudimentary testing of Gaussian filtering.
"""
import unittest

import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import gaussianfilter
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace.discrete.discretegaussianmodel import DiscreteGaussianLinearModel

TEST_NDIM = 2


class MockGaussianFilter(gaussianfilter.GaussianFilter):
    """
    """

    def __init__(self, dynamod, measmod, initdist):
        """
        """
        self.dynamod = dynamod
        self.measmod = measmod
        self.initdist = initdist

    def predict(self, start, stop, randvar, *args, **kwargs):
        return randvar

    def update(self, time, randvar, data, *args, **kwargs):
        return randvar, None, None, None

    @property
    def dynamicmodel(self):
        return self.dynamod

    @property
    def measurementmodel(self):
        return self.measmod

    @property
    def initialdistribution(self):
        return self.initdist


class TestGaussianFilter(unittest.TestCase):
    """
    Check whether with a mock object
    (minimal implementation) the filter()
    method runs well.
    """

    def setUp(self):
        """
        """
        mtrx = np.random.rand(TEST_NDIM, TEST_NDIM)
        self.mockmod = DiscreteGaussianLinearModel(lambda t: mtrx, lambda t: 0,
                                          lambda t: mtrx @ mtrx.T)
        mean, cov = np.ones(TEST_NDIM), np.eye(TEST_NDIM)
        self.initdist = RandomVariable(distribution=Normal(mean, cov))
        self.mgf = MockGaussianFilter(self.mockmod, self.mockmod,
                                      self.initdist)

    def test_filter(self):
        """
        Note that the output of the filter for given
        predict() and update() is meaningless.
        This test is all about making sure the function
        returns something of the right shape.
        """
        start = np.random.rand()
        stop = start + 0.05 + 2 * np.random.rand()
        step = 0.05 * np.random.rand()
        timesteps = np.arange(start, stop, step)
        means, covars = self.mgf.filter_stream(
            lambda t: np.random.rand(TEST_NDIM), timesteps)
        self.assertEqual(means.ndim, 2)
        self.assertEqual(means.shape[0], timesteps.shape[0])
        self.assertEqual(means.shape[1], TEST_NDIM)
        self.assertEqual(covars.ndim, 3)
        self.assertEqual(covars.shape[0], timesteps.shape[0])
        self.assertEqual(covars.shape[1], TEST_NDIM)
        self.assertEqual(covars.shape[2], TEST_NDIM)

    def test_dynamicmodel(self):
        """
        """
        self.assertEqual(self.mgf.dynamicmodel, self.mockmod)

    def test_measurementmodel(self):
        """
        """
        self.assertEqual(self.mgf.measurementmodel, self.mockmod)

    def test_initialdistribution(self):
        """
        """
        self.assertEqual(self.mgf.initialdistribution, self.initdist)
