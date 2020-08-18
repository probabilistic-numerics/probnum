import unittest

import numpy as np

from probnum.filtsmooth.statespace.discrete import discretegaussianmodel

TEST_NDIM = 4


class TestDiscreteGaussianModel(unittest.TestCase):
    def setUp(self):
        dynamat = np.random.rand(TEST_NDIM, TEST_NDIM)
        diffmat = dynamat @ dynamat.T + np.eye(TEST_NDIM)
        self.nl = discretegaussianmodel.DiscreteGaussianModel(
            lambda t, x: dynamat @ x, lambda t: diffmat
        )

    def test_dynamics(self):
        some_input = np.random.rand(TEST_NDIM)
        val = self.nl.dynamics(0.0, some_input)
        self.assertEqual(val.ndim, 1)
        self.assertEqual(val.shape[0], TEST_NDIM)

    def test_diffusionmatrix(self):
        val = self.nl.diffusionmatrix(0.0)
        self.assertEqual(val.ndim, 2)
        self.assertEqual(val.shape[0], TEST_NDIM)
        self.assertEqual(val.shape[1], TEST_NDIM)

    def test_jacobian(self):
        some_input = np.random.rand(TEST_NDIM)
        with self.assertRaises(NotImplementedError):
            self.nl.jacobian(0.0, some_input)

    def test_sample(self):
        some_input = np.random.rand(TEST_NDIM)
        samp = self.nl.sample(0.0, some_input)
        self.assertEqual(samp.ndim, 1)
        self.assertEqual(samp.shape[0], TEST_NDIM)

    def test_ndim(self):
        self.assertEqual(self.nl.ndim, TEST_NDIM)

    def test_pdf(self):
        some_state = np.random.rand(TEST_NDIM)
        evl = self.nl.pdf(some_state, 0.0, some_state)
        self.assertEqual(np.isscalar(evl), True)


class TestLinear(unittest.TestCase):
    def setUp(self):
        dynamat = np.random.rand(TEST_NDIM, TEST_NDIM)
        diffmat = dynamat @ dynamat.T + np.eye(TEST_NDIM)
        self.lin = discretegaussianmodel.DiscreteGaussianLinearModel(
            lambda t: dynamat, lambda t: np.random.rand(TEST_NDIM), lambda t: diffmat
        )

    def test_dynamics(self):
        some_input = np.random.rand(TEST_NDIM)
        val = self.lin.dynamics(0.0, some_input)
        self.assertEqual(val.ndim, 1)
        self.assertEqual(val.shape[0], TEST_NDIM)

    def test_diffusionmatrix(self):
        val = self.lin.diffusionmatrix(0.0)
        self.assertEqual(val.ndim, 2)
        self.assertEqual(val.shape[0], TEST_NDIM)
        self.assertEqual(val.shape[1], TEST_NDIM)

    def test_jacobian(self):
        some_input = np.random.rand(TEST_NDIM)
        jac = self.lin.jacobian(0.0, some_input)
        self.assertEqual(jac.ndim, 2)
        self.assertEqual(jac.shape[0], TEST_NDIM)
        self.assertEqual(jac.shape[1], TEST_NDIM)

    def test_dynamicsmatrix(self):
        dyna = self.lin.dynamicsmatrix(0.0)
        self.assertEqual(dyna.ndim, 2)
        self.assertEqual(dyna.shape[0], TEST_NDIM)
        self.assertEqual(dyna.shape[1], TEST_NDIM)

    def test_force(self):
        force = self.lin.force(0.0)
        self.assertEqual(force.ndim, 1)
        self.assertEqual(force.shape[0], TEST_NDIM)

    def test_sample(self):
        some_input = np.random.rand(TEST_NDIM)
        samp = self.lin.sample(0.0, some_input)
        self.assertEqual(samp.ndim, 1)
        self.assertEqual(samp.shape[0], TEST_NDIM)

    def test_ndim(self):
        self.assertEqual(self.lin.ndim, TEST_NDIM)

    def test_pdf(self):
        some_state = np.random.rand(TEST_NDIM)
        evl = self.lin.pdf(some_state, 0.0, some_state)
        self.assertEqual(np.isscalar(evl), True)


class TestLTI(unittest.TestCase):
    def setUp(self):
        dynamat = np.random.rand(TEST_NDIM, TEST_NDIM)
        diffmat = dynamat @ dynamat.T + np.eye(TEST_NDIM)
        self.lti = discretegaussianmodel.DiscreteGaussianLTIModel(
            dynamat, dynamat[0], diffmat
        )

    def test_dynamics(self):
        some_input = np.random.rand(TEST_NDIM)
        val = self.lti.dynamics(0.0, some_input)
        self.assertEqual(val.ndim, 1)
        self.assertEqual(val.shape[0], TEST_NDIM)

    def test_dynamicsmatrix(self):
        some_input = np.random.rand(TEST_NDIM)
        dyna = self.lti.dynamicsmatrix(0.0)
        self.assertEqual(dyna.ndim, 2)
        self.assertEqual(dyna.shape[0], TEST_NDIM)
        self.assertEqual(dyna.shape[1], TEST_NDIM)

    def test_diffusionmatrix(self):
        val = self.lti.diffusionmatrix(0.0)
        self.assertEqual(val.ndim, 2)
        self.assertEqual(val.shape[0], TEST_NDIM)
        self.assertEqual(val.shape[1], TEST_NDIM)

    def test_jacobian(self):
        some_input = np.random.rand(TEST_NDIM)
        jac = self.lti.jacobian(0.0, some_input)
        self.assertEqual(jac.ndim, 2)
        self.assertEqual(jac.shape[0], TEST_NDIM)
        self.assertEqual(jac.shape[1], TEST_NDIM)

    def test_force(self):
        force = self.lti.force(0.0)
        self.assertEqual(force.ndim, 1)
        self.assertEqual(force.shape[0], TEST_NDIM)

    def test_sample(self):
        some_input = np.random.rand(TEST_NDIM)
        samp = self.lti.sample(0.0, some_input)
        self.assertEqual(samp.ndim, 1)
        self.assertEqual(samp.shape[0], TEST_NDIM)

    def test_ndim(self):
        self.assertEqual(self.lti.ndim, TEST_NDIM)

    def test_pdf(self):
        some_state = np.random.rand(TEST_NDIM)
        evl = self.lti.pdf(some_state, 0.0, some_state)
        self.assertEqual(np.isscalar(evl), True)
