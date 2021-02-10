import warnings

import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss
from tests.testing import NumpyAssertions

from .test_transition import InterfaceTestTransition


class TestSDE(InterfaceTestTransition):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1):

        self.g = lambda t, x: np.sin(x)
        self.L = lambda t: spdmat1
        self.dg = lambda t, x: np.cos(x)
        self.transition = pnfss.SDE(test_ndim, self.g, self.L, self.dg)

    # Test access to system matrices

    def test_drift(self, some_normal_rv1):
        expected = self.g(0.0, some_normal_rv1.mean)
        received = self.transition.driftfun(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_dispersionmatrix(self):
        expected = self.L(0.0)
        received = self.transition.dispmatfun(0.0)
        np.testing.assert_allclose(received, expected)

    def test_jacobfun(self, some_normal_rv1):
        expected = self.dg(0.0, some_normal_rv1.mean)
        received = self.transition.jacobfun(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_rv(some_normal_rv1, 0.0, dt=0.1)

    def test_forward_realization(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_realization(some_normal_rv1.sample(), 0.0, dt=0.1)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2, 0.0, dt=0.1)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(
                some_normal_rv1.sample(), some_normal_rv2, 0.0, dt=0.1
            )

    def test_input_dim(self, test_ndim):
        assert self.transition.input_dim == test_ndim

    def test_output_dim(self, test_ndim):
        assert self.transition.output_dim == test_ndim

    def test_dimension(self, test_ndim):
        assert self.transition.dimension == test_ndim


class TestLinearSDE(TestSDE):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1, spdmat2):

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2
        self.transition = pnfss.LinearSDE(test_ndim, self.G, self.v, self.L)

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    def test_driftmatfun(self):
        expected = self.G(0.0)
        received = self.transition.driftmatfun(0.0)
        np.testing.assert_allclose(expected, received)

    def test_forcevecfun(self):
        expected = self.v(0.0)
        received = self.transition.forcevecfun(0.0)
        np.testing.assert_allclose(expected, received)

    def test_forward_rv(self, some_normal_rv1):
        out, _ = self.transition.forward_rv(some_normal_rv1, t=0.0, dt=0.1)
        assert isinstance(out, pnrv.Normal)

    def test_forward_realization(self, some_normal_rv1):
        out, info = self.transition.forward_realization(
            some_normal_rv1.sample(), t=0.0, dt=0.1
        )
        assert isinstance(out, pnrv.Normal)
        warnings.warn(
            "\n\n\n Check that the resulting values of forward "
            "are correct by comparing to MFD?!\n\n\n"
        )

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2, t=0.0, dt=0.1)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(
                some_normal_rv1.sample(), some_normal_rv2, t=0.0, dt=0.1
            )


@pytest.fixture(params=["classic", "sqrt"])
def forw_impl_string_linear_gauss(request):
    """Forward implementation choices passed via strings."""
    return request.param


@pytest.fixture(params=["classic", "joseph", "sqrt"])
def backw_impl_string_linear_gauss(request):
    """Backward implementation choices passed via strings."""
    return request.param


class TestLTISDE(TestLinearSDE):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.G_const = spdmat1
        self.v_const = np.arange(test_ndim)
        self.L_const = spdmat2

        self.transition = pnfss.LTISDE(
            self.G_const,
            self.v_const,
            self.L_const,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    def test_discretise(self):
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, pnfss.DiscreteLTIGaussian)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(
            some_normal_rv1, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, pnrv.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.sample(), some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, pnrv.Normal)


#
# class TestLTISDE(unittest.TestCase, NumpyAssertions):
#
#     start = np.random.rand()
#     stop = start + np.random.rand()
#     some_rv = pnrv.Normal(np.random.rand(TEST_NDIM), np.eye(TEST_NDIM))
#     some_nongaussian_rv = pnrv.Constant(np.random.rand(TEST_NDIM))
#
#     def setUp(self) -> None:
#
#         self.F = np.random.rand(TEST_NDIM, TEST_NDIM)
#         self.s = np.zeros(TEST_NDIM)  # only because MFD is lazy so far
#         self.L = np.random.rand(TEST_NDIM, 2)
#         self.sde = pnfss.sde.LTISDE(driftmat=self.F, forcevec=self.s, dispmat=self.L)
#
#     def test_driftmatrix(self):
#         self.assertAllClose(self.sde.driftmat, self.F)
#
#     def test_force(self):
#         self.assertAllClose(self.sde.forcevec, self.s)
#
#     def test_dispersionmatrix(self):
#
#         self.assertAllClose(self.sde.dispmat, self.L)
#
#     def test_discretise(self):
#         discrete = self.sde.discretise(dt=self.stop - self.start)
#         self.assertIsInstance(discrete, pnfss.discrete_transition.DiscreteLTIGaussian)
#
#     def test_transition_rv(self):
#
#         with self.subTest("Output reachable"):
#             _ = self.sde.forward_rv(self.some_rv, self.start, dt=self.stop - self.start)
#
#     def test_transition_realization(self):
#
#         with self.subTest("Output reachable"):
#             _ = self.sde.forward_realization(
#                 self.some_rv.sample(),
#                 self.start,
#                 dt=self.stop - self.start,
#             )
#
#
# class TestLinearSDEStatistics(unittest.TestCase, NumpyAssertions):
#     """Test against Matrix Fraction decomposition."""
#
#     start = 0.1
#     stop = start + 0.1
#     some_rv = pnrv.Normal(
#         np.random.rand(TEST_NDIM), np.diag(1 + np.random.rand(TEST_NDIM))
#     )
#     step = (stop - start) / 20.0
#
#     def setUp(self):
#         self.Fmat = np.random.rand(TEST_NDIM, TEST_NDIM)
#         self.svec = np.zeros(TEST_NDIM)  # only because MFD is lazy so far
#         self.Lmat = np.random.rand(TEST_NDIM, 2)
#
#         def f(t, x):
#             return self.Fmat @ x
#
#         def df(t):
#             return self.Fmat
#
#         def L(t):
#             return self.Lmat
#
#         self.f = f
#         self.df = df
#         self.L = L
#
#     def test_linear_sde_statistics(self):
#         out_rv, _ = pnfss.sde.solve_moment_equations_forward(
#             self.some_rv, self.start, self.stop, self.step, self.f, self.df, self.L
#         )
#         ah, qh, _ = pnfss.sde.matrix_fraction_decomposition(
#             self.Fmat, self.Lmat, self.stop - self.start
#         )
#
#         self.assertAllClose(out_rv.mean, ah @ self.some_rv.mean, rtol=1e-6)
#         self.assertAllClose(out_rv.cov, ah @ self.some_rv.cov @ ah.T + qh, rtol=1e-6)
#
#
# class TestMatrixFractionDecomposition(unittest.TestCase):
#     """Test MFD against closed-form IBM solution."""
#
#     def setUp(self):
#         self.a = np.array([[0, 1], [0, 0]])
#         self.dc = 1.23451432151241
#         self.b = self.dc * np.array([[0], [1]])
#         self.h = 0.1
#
#     def test_ibm_qh_stack(self):
#         *_, stack = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
#
#         with self.subTest("top left"):
#             error = np.linalg.norm(stack[:2, :2] - self.a)
#             self.assertLess(error, 1e-15)
#
#         with self.subTest("top right"):
#             error = np.linalg.norm(stack[:2, 2:] - self.b @ self.b.T)
#             self.assertLess(error, 1e-15)
#
#         with self.subTest("bottom left"):
#             error = np.linalg.norm(stack[2:, 2:] + self.a.T)
#             self.assertLess(error, 1e-15)
#
#         with self.subTest("bottom right"):
#             error = np.linalg.norm(stack[2:, :2] - 0.0)
#             self.assertLess(error, 1e-15)
#
#     def test_ibm_ah(self):
#         Ah, *_ = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
#         expected = np.array([[1, self.h], [0, 1]])
#         error = np.linalg.norm(Ah - expected)
#         self.assertLess(error, 1e-15)
#
#     def test_ibm_qh(self):
#         _, Qh, _ = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
#         expected = self.dc ** 2 * np.array(
#             [[self.h ** 3 / 3, self.h ** 2 / 2], [self.h ** 2 / 2, self.h]]
#         )
#         error = np.linalg.norm(Qh - expected)
#         self.assertLess(error, 1e-15)
#
#     def test_type_error_captured(self):
#         good_A = np.array([[0, 1], [0, 0]])
#         good_B = np.array([[0], [1]])
#         good_h = 0.1
#         with self.subTest(culprit="F"):
#             with self.assertRaises(ValueError):
#                 pnfss.sde.matrix_fraction_decomposition(
#                     np.random.rand(2), good_B, good_h
#                 )
#
#         with self.subTest(culprit="L"):
#             with self.assertRaises(ValueError):
#                 pnfss.sde.matrix_fraction_decomposition(
#                     good_A, np.random.rand(2), good_h
#                 )
#
#         with self.subTest(culprit="h"):
#             with self.assertRaises(ValueError):
#                 pnfss.sde.matrix_fraction_decomposition(
#                     good_A, good_B, np.random.rand(2)
#                 )
