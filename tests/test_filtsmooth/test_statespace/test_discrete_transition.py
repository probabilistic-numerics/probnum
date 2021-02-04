import unittest

import numpy as np

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv
from tests.testing import NumpyAssertions

TEST_NDIM = 4


class TestDiscreteGaussianTransition(unittest.TestCase, NumpyAssertions):

    some_rv = pnrv.Normal(
        mean=np.random.rand(TEST_NDIM), cov=np.diag(1 + np.random.rand(TEST_NDIM))
    )
    start = 0.1 + 0.01 * np.random.rand()

    def setUp(self):
        def g(t, x):
            return np.sin(x)

        def S(t):
            return np.eye(TEST_NDIM)

        def dg(t, x):
            return np.cos(x)

        self.dtrans = pnfss.discrete_transition.DiscreteGaussian(g, S, dg)

        self.g = g
        self.S = S
        self.dg = dg

    def test_dynamics(self):
        received = self.dtrans.state_trans_fun(self.start, self.some_rv.mean)
        expected = self.g(self.start, self.some_rv.mean)
        self.assertAllClose(received, expected)

    def test_diffusionmatrix(self):
        received = self.dtrans.diffmatfun(self.start)
        expected = self.S(self.start)
        self.assertAllClose(received, expected)

    def test_jacobian(self):
        received = self.dtrans.jacobfun(self.start, self.some_rv.mean)
        expected = self.dg(self.start, self.some_rv.mean)
        self.assertAllClose(received, expected)

    def test_jacobian_error(self):
        """Calling a Jacobian when nothing is provided throws an Exception."""
        dtrans_no_jacob = pnfss.discrete_transition.DiscreteGaussian(self.g, self.S)
        with self.assertRaises(NotImplementedError):
            dtrans_no_jacob.jacobfun(self.start, self.some_rv.mean)

    def test_transition_rv(self):

        with self.assertRaises(NotImplementedError):
            self.dtrans.transition_rv(self.some_rv, self.start)

    def test_transition_realization(self):
        self.dtrans.transition_realization(self.some_rv.sample(), self.start)


class TestDiscreteLinearGaussianTransition(unittest.TestCase, NumpyAssertions):

    some_rv = pnrv.Normal(
        mean=np.random.rand(TEST_NDIM), cov=np.diag(1 + np.random.rand(TEST_NDIM))
    )
    some_nongaussian_rv = pnrv.Constant(np.random.rand(TEST_NDIM))
    start = 0.1 + 0.01 * np.random.rand()

    def setUp(self):
        def G(t):
            return np.arange(TEST_NDIM ** 2).reshape((TEST_NDIM, TEST_NDIM))

        def v(t):
            return np.ones(TEST_NDIM)

        def S(t):
            return np.eye(TEST_NDIM)

        self.dtrans = pnfss.discrete_transition.DiscreteLinearGaussian(G, v, S)

        self.G = G
        self.v = v
        self.S = S

    def test_transition_rv(self):

        with self.subTest("Non-Normal-RV-exception"):
            with self.assertRaises(TypeError):
                self.dtrans.transition_rv(self.some_nongaussian_rv, self.start)

    def test_dynamicsmatrix(self):
        received = self.dtrans.dynamicsmatfun(self.start)
        expected = self.G(self.start)
        self.assertAllClose(received, expected)

    def test_forcevector(self):
        received = self.dtrans.forcevecfun(self.start)
        expected = self.v(self.start)
        self.assertAllClose(received, expected)

    def test_dimension(self):
        self.assertEqual(self.dtrans.dimension, TEST_NDIM)


class TestDiscreteLTIGaussianTransition(unittest.TestCase, NumpyAssertions):
    def test_init_exceptions(self):

        good_dynamicsmat = np.ones((TEST_NDIM, 4 * TEST_NDIM))
        good_forcevec = np.ones(TEST_NDIM)
        good_diffmat = np.ones((TEST_NDIM, TEST_NDIM))

        with self.subTest("Baseline should work"):
            pnfss.discrete_transition.DiscreteLinearGaussian(
                good_dynamicsmat, good_forcevec, good_diffmat
            )

        with self.subTest("bad dynamics"):
            with self.assertRaises(TypeError):
                pnfss.discrete_transition.DiscreteLTIGaussian(
                    good_forcevec, good_forcevec, good_diffmat
                )

        with self.subTest("bad force"):
            with self.assertRaises(TypeError):
                pnfss.discrete_transition.DiscreteLTIGaussian(
                    good_dynamicsmat, good_dynamicsmat, good_diffmat
                )

        with self.subTest("bad diffusion"):
            with self.assertRaises(TypeError):
                pnfss.discrete_transition.DiscreteLTIGaussian(
                    good_dynamicsmat, good_forcevec, good_dynamicsmat
                )
