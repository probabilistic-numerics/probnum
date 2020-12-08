import unittest

import numpy as np

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv
from tests.testing import NumpyAssertions

TEST_NDIM = 10


class TestSDE(unittest.TestCase, NumpyAssertions):

    start = np.random.rand()
    stop = start + np.random.rand()
    some_rv = pnrv.Normal(np.random.rand(TEST_NDIM), np.eye(TEST_NDIM))

    def setUp(self) -> None:
        def f(t, x):
            return x + 1.0

        def l(t):
            return 2.0

        def df(t, x):
            return np.eye(len(x)) + 3

        self.sde = pnfss.sde.SDE(driftfun=f, dispmatfun=l, jacobfun=df)
        self.f = f
        self.l = l
        self.df = df

    def test_transition_realization(self):
        with self.assertRaises(NotImplementedError):
            self.sde.transition_realization(
                self.some_rv.sample(), self.start, self.stop
            )

    def test_transition_rv(self):
        with self.assertRaises(NotImplementedError):
            self.sde.transition_rv(self.some_rv, self.start, self.stop)

    def test_drift(self):
        expected = self.f(self.start, self.some_rv.mean)
        received = self.sde.driftfun(self.start, self.some_rv.mean)
        self.assertAllClose(received, expected)

    def test_dispersionmatrix(self):
        expected = self.l(self.start)
        received = self.sde.dispmatfun(self.start)
        self.assertAllClose(received, expected)

    def test_jacobfun(self):
        expected = self.df(self.start, self.some_rv.mean)
        received = self.sde.jacobfun(self.start, self.some_rv.mean)
        self.assertAllClose(received, expected)

    def test_dimension(self):
        with self.assertRaises(NotImplementedError):
            _ = self.sde.dimension


class TestLinearSDE(unittest.TestCase, NumpyAssertions):

    start = np.random.rand()
    stop = start + np.random.rand()
    some_rv = pnrv.Normal(np.random.rand(TEST_NDIM), np.eye(TEST_NDIM))
    some_nongaussian_rv = pnrv.Constant(np.random.rand(TEST_NDIM))
    rk_step = (stop - start) / 10.0

    def setUp(self) -> None:
        def F(t):
            return 1 + np.arange(TEST_NDIM ** 2).reshape((TEST_NDIM, TEST_NDIM))

        def s(t):
            return 1 + np.arange(TEST_NDIM)

        def L(t):
            return 1 + np.arange(2 * TEST_NDIM).reshape((TEST_NDIM, 2))

        self.sde = pnfss.sde.LinearSDE(driftmatfun=F, forcevecfun=s, dispmatfun=L)
        self.F = F
        self.s = s
        self.L = L

    def test_transition_realization(self):

        _ = self.sde.transition_realization(
            self.some_rv.sample(), self.start, self.stop, step=self.rk_step
        )

    def test_transition_rv(self):

        with self.assertRaises(TypeError):
            self.sde.transition_rv(
                self.some_nongaussian_rv,
                self.start,
                self.stop,
                step=self.rk_step,
            )

        with self.subTest("Output attainable"):
            _ = self.sde.transition_rv(
                self.some_rv, self.start, self.stop, step=self.rk_step
            )

    def test_dimension(self):
        self.assertEqual(self.sde.dimension, TEST_NDIM)


class TestLTISDE(unittest.TestCase, NumpyAssertions):

    start = np.random.rand()
    stop = start + np.random.rand()
    some_rv = pnrv.Normal(np.random.rand(TEST_NDIM), np.eye(TEST_NDIM))
    some_nongaussian_rv = pnrv.Constant(np.random.rand(TEST_NDIM))

    def setUp(self) -> None:

        self.F = np.random.rand(TEST_NDIM, TEST_NDIM)
        self.s = np.zeros(TEST_NDIM)  # only because MFD is lazy so far
        self.L = np.random.rand(TEST_NDIM, 2)
        self.sde = pnfss.sde.LTISDE(driftmat=self.F, forcevec=self.s, dispmat=self.L)

    def test_driftmatrix(self):
        self.assertAllClose(self.sde.driftmat, self.F)

    def test_force(self):
        self.assertAllClose(self.sde.forcevec, self.s)

    def test_dispersionmatrix(self):

        self.assertAllClose(self.sde.dispmat, self.L)

    def test_discretise(self):
        discrete = self.sde.discretise(step=self.stop - self.start)
        self.assertIsInstance(discrete, pnfss.discrete_transition.DiscreteLTIGaussian)

    def test_transition_rv(self):

        with self.subTest("NormalRV exception"):
            with self.assertRaises(TypeError):
                self.sde.transition_rv(
                    self.some_nongaussian_rv,
                    self.start,
                    self.stop,
                )

        with self.subTest("Output attainable"):
            _ = self.sde.transition_rv(self.some_rv, self.start, self.stop)

    def test_transition_realization(self):

        with self.subTest("Output attainable"):
            _ = self.sde.transition_realization(
                self.some_rv.sample(),
                self.start,
                self.stop,
            )


class TestLinearSDEStatistics(unittest.TestCase, NumpyAssertions):
    """Test against Matrix Fraction decomposition."""

    start = 0.1
    stop = start + 0.1
    some_rv = pnrv.Normal(
        np.random.rand(TEST_NDIM), np.diag(1 + np.random.rand(TEST_NDIM))
    )
    step = (stop - start) / 20.0

    def setUp(self):
        self.Fmat = np.random.rand(TEST_NDIM, TEST_NDIM)
        self.svec = np.zeros(TEST_NDIM)  # only because MFD is lazy so far
        self.Lmat = np.random.rand(TEST_NDIM, 2)

        def f(t, x):
            return self.Fmat @ x

        def df(t):
            return self.Fmat

        def L(t):
            return self.Lmat

        self.f = f
        self.df = df
        self.L = L

    def test_linear_sde_statistics(self):
        out_rv, _ = pnfss.sde.linear_sde_statistics(
            self.some_rv, self.start, self.stop, self.step, self.f, self.df, self.L
        )
        ah, qh, _ = pnfss.sde.matrix_fraction_decomposition(
            self.Fmat, self.Lmat, self.stop - self.start
        )

        self.assertAllClose(out_rv.mean, ah @ self.some_rv.mean, rtol=1e-6)
        self.assertAllClose(out_rv.cov, ah @ self.some_rv.cov @ ah.T + qh, rtol=1e-6)


class TestMatrixFractionDecomposition(unittest.TestCase):
    """Test MFD against closed-form IBM solution."""

    def setUp(self):
        self.a = np.array([[0, 1], [0, 0]])
        self.dc = 1.23451432151241
        self.b = self.dc * np.array([[0], [1]])
        self.h = 0.1

    def test_ibm_qh_stack(self):
        *_, stack = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)

        with self.subTest("top left"):
            error = np.linalg.norm(stack[:2, :2] - self.a)
            self.assertLess(error, 1e-15)

        with self.subTest("top right"):
            error = np.linalg.norm(stack[:2, 2:] - self.b @ self.b.T)
            self.assertLess(error, 1e-15)

        with self.subTest("bottom left"):
            error = np.linalg.norm(stack[2:, 2:] + self.a.T)
            self.assertLess(error, 1e-15)

        with self.subTest("bottom right"):
            error = np.linalg.norm(stack[2:, :2] - 0.0)
            self.assertLess(error, 1e-15)

    def test_ibm_ah(self):
        Ah, *_ = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
        expected = np.array([[1, self.h], [0, 1]])
        error = np.linalg.norm(Ah - expected)
        self.assertLess(error, 1e-15)

    def test_ibm_qh(self):
        _, Qh, _ = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
        expected = self.dc ** 2 * np.array(
            [[self.h ** 3 / 3, self.h ** 2 / 2], [self.h ** 2 / 2, self.h]]
        )
        error = np.linalg.norm(Qh - expected)
        self.assertLess(error, 1e-15)

    def test_type_error_captured(self):
        good_A = np.array([[0, 1], [0, 0]])
        good_B = np.array([[0], [1]])
        good_h = 0.1
        with self.subTest(culprit="F"):
            with self.assertRaises(ValueError):
                pnfss.sde.matrix_fraction_decomposition(
                    np.random.rand(2), good_B, good_h
                )

        with self.subTest(culprit="L"):
            with self.assertRaises(ValueError):
                pnfss.sde.matrix_fraction_decomposition(
                    good_A, np.random.rand(2), good_h
                )

        with self.subTest(culprit="h"):
            with self.assertRaises(ValueError):
                pnfss.sde.matrix_fraction_decomposition(
                    good_A, good_B, np.random.rand(2)
                )
