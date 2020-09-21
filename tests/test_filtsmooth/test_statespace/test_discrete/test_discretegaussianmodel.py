import unittest

import numpy as np

from probnum.filtsmooth.statespace.discrete import discretegaussianmodel
import probnum.random_variables as rvs

TEST_NDIM = 4


class TestDiscreteGaussianModel(unittest.TestCase):
    def setUp(self):
        dynamat = np.random.rand(TEST_NDIM, TEST_NDIM)
        diffmat = dynamat @ dynamat.T + np.eye(TEST_NDIM)
        self.nl = discretegaussianmodel.DiscreteGaussianModel(
            lambda t, x: dynamat @ x, lambda t: diffmat
        )

    def test_transition_rv(self):
        with self.assertRaises(NotImplementedError):
            self.nl.transition_rv(
                rvs.Normal(np.ones(self.nl.dimension), np.eye(self.nl.dimension)), start=None
            )

    def test_transition_realization(self):
        out_rv = self.nl.transition_realization(np.ones(self.nl.dimension), start=None)
        self.assertIsInstance(out_rv, rvs.RandomVariable)

    def test_dimension(self):
        self.assertEqual(self.nl.dimension, TEST_NDIM)


class TestLinear(TestDiscreteGaussianModel):
    def setUp(self):
        dynamat = np.random.rand(TEST_NDIM, TEST_NDIM)
        diffmat = dynamat @ dynamat.T + np.eye(TEST_NDIM)
        self.nl = discretegaussianmodel.DiscreteGaussianLinearModel(
            lambda t: dynamat, lambda t: np.random.rand(TEST_NDIM), lambda t: diffmat
        )

    def test_dynamicsmatrix(self):
        dyna = self.nl.dynamicsmatrix(0.0)
        self.assertEqual(dyna.ndim, 2)
        self.assertEqual(dyna.shape[0], TEST_NDIM)
        self.assertEqual(dyna.shape[1], TEST_NDIM)

    def test_forcevector(self):
        force = self.nl.forcevector(0.0)
        self.assertEqual(force.ndim, 1)
        self.assertEqual(force.shape[0], TEST_NDIM)

    def test_transition_rv(self):
        out_rv = self.nl.transition_rv(
            rvs.Normal(np.ones(self.nl.dimension), np.eye(self.nl.dimension)), start=None
        )
        self.assertIsInstance(out_rv, rvs.RandomVariable)


class TestLTI(TestLinear):
    def setUp(self):
        dynamat = np.random.rand(TEST_NDIM, TEST_NDIM)
        diffmat = dynamat @ dynamat.T + np.eye(TEST_NDIM)
        self.nl = discretegaussianmodel.DiscreteGaussianLTIModel(
            dynamat, dynamat[0], diffmat
        )
