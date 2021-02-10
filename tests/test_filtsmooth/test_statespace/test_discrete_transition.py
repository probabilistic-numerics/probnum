import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss

from .test_transition import InterfaceTestTransition


class TestDiscreteGaussian(InterfaceTestTransition):
    """Tests for the discrete Gaussian class.

    Some are reused/overwritten for subclasses, therefore the class-
    structure.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1):

        self.g = lambda t, x: np.sin(x)
        self.S = lambda t: spdmat1
        self.dg = lambda t, x: np.cos(x)
        self.transition = pnfss.DiscreteGaussian(
            test_ndim, test_ndim, self.g, self.S, self.dg
        )

    def test_state_transition(self, some_normal_rv1):
        received = self.transition.state_trans_fun(0.0, some_normal_rv1.mean)
        expected = self.g(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_process_noise(self):
        received = self.transition.proc_noise_cov_mat_fun(0.0)
        expected = self.S(0.0)
        np.testing.assert_allclose(received, expected)

    def test_process_noise_cholesky(self):
        received = self.transition.proc_noise_cov_cholesky_fun(0.0)
        expected = np.linalg.cholesky(self.transition.proc_noise_cov_mat_fun(0.0))
        np.testing.assert_allclose(received, expected)

    def test_jacobian(self, some_normal_rv1):
        received = self.transition.jacob_state_trans_fun(0.0, some_normal_rv1.mean)
        expected = self.dg(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_forward_rv(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_rv(some_normal_rv1, 0.0)

    def test_forward_realization(self, some_normal_rv1):
        out, _ = self.transition.forward_realization(some_normal_rv1.sample(), 0.0)
        assert isinstance(out, pnrv.Normal)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(
                some_normal_rv1.sample(), some_normal_rv2
            )

    def test_input_dim(self, test_ndim):
        assert self.transition.input_dim == test_ndim

    def test_output_dim(self, test_ndim):
        assert self.transition.output_dim == test_ndim


class TestLinearGaussian(TestDiscreteGaussian):
    """Test class for linear Gaussians. Inherits tests from `TestDiscreteGaussian` but
    overwrites the forward and backward transitions.

    Also tests that different forward and backward implementations yield
    the same results.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
        spdmat1,
        spdmat2,
        forw_impl_string="classic",
        backw_impl_string="classic",
    ):

        self.G = lambda t: spdmat1
        self.S = lambda t: spdmat2
        self.v = lambda t: np.arange(test_ndim)
        self.transition = pnfss.DiscreteLinearGaussian(
            test_ndim,
            test_ndim,
            self.G,
            self.v,
            self.S,
            forward_implementation=forw_impl_string,
            backward_implementation=backw_impl_string,
        )

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    def test_forward_rv(self, some_normal_rv1):
        out, _ = self.transition.forward_rv(some_normal_rv1, 0.0)
        assert isinstance(out, pnrv.Normal)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(some_normal_rv1, some_normal_rv2)
        assert isinstance(out, pnrv.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.sample(), some_normal_rv2
        )
        assert isinstance(out, pnrv.Normal)


# import unittest
#
# import numpy as np
#
# import probnum.filtsmooth.statespace as pnfss
# import probnum.random_variables as pnrv
# from probnum.problems.zoo.linalg import random_spd_matrix
# from tests.testing import NumpyAssertions
#
# TEST_NDIM = 4
# import pytest
#
# # Tests for discrete Gaussian transitions
#
#
# @pytest.fixture
# def test_ndim():
#     return 4
#
#
# @pytest.fixture
# def spdmat1(test_ndim):
#     return random_spd_matrix(test_ndim)
#
#
# @pytest.fixture
# def spdmat2(test_ndim):
#     return random_spd_matrix(test_ndim)
#
#
# @pytest.fixture
# def g():
#     return lambda t, x: np.sin(x)
#
#
# @pytest.fixture
# def S(spdmat1):
#     return lambda t: spdmat1
#
#
# @pytest.fixture
# def dg():
#     return lambda t, x: np.cos(x)
#
#
# @pytest.fixture
# def discrete_transition(g, S, dg):
#     return pnfss.discrete_transition.DiscreteGaussian(g, S, dg)
#
#
# @pytest.fixture
# def some_rv(test_ndim, spdmat2):
#     return pnrv.Normal(mean=np.random.rand(test_ndim), cov=spdmat2)
#
#
# def test_state_transition(discrete_transition, some_rv, g):
#     received = discrete_transition.state_trans_fun(0.0, some_rv.mean)
#     expected = g(0.0, some_rv.mean)
#     np.testing.assert_allclose(received, expected)
#
#
# def test_proces_noise(discrete_transition, some_rv, S):
#     received = discrete_transition.proc_noise_cov_mat_fun(0.0)
#     expected = S(0.0)
#     np.testing.assert_allclose(received, expected)
#
#
# def test_jacobian(discrete_transition, some_rv, dg):
#     received = discrete_transition.jacob_state_trans_fun(0.0, some_rv.mean)
#     expected = dg(0.0, some_rv.mean)
#     np.testing.assert_allclose(received, expected)
#
#
# def test_jacobian_exception(g, S, some_rv):
#     """Calling a Jacobian when nothing is provided throws an Exception."""
#     dtrans_no_jacob = pnfss.discrete_transition.DiscreteGaussian(g, S)
#     with pytest.raises(NotImplementedError):
#         dtrans_no_jacob.jacob_state_trans_fun(0.0, some_rv.mean)
#
#
# def test_forward_rv_nonlinear(discrete_transition, some_rv):
#     with pytest.raises(NotImplementedError):
#         discrete_transition.forward_rv(some_rv, 0.0)
#
#
# def test_backward_rv_nonlinear(discrete_transition, some_rv):
#     with pytest.raises(NotImplementedError):
#         discrete_transition.backward_rv(some_rv, some_rv)
#
#
# def test_backward_realization(discrete_transition, some_rv):
#     with pytest.raises(NotImplementedError):
#         discrete_transition.backward_realization(some_rv, some_rv)
#
#
# def test_forward_realization(discrete_transition, some_rv):
#     out, _ = discrete_transition.forward_realization(some_rv.sample(), 0.0)
#     assert isinstance(out, pnrv.Normal)
#
#
# def test_proc_noise_cov_cholesky_fun(discrete_transition):
#     expected = np.linalg.cholesky(discrete_transition.proc_noise_cov_mat_fun(0))
#     received = discrete_transition.proc_noise_cov_cholesky_fun(0)
#     np.testing.assert_allclose(expected, received)
#
#
# # Tests for discrete, linear Gaussian transitions
#
#
# @pytest.fixture
# def G(spdmat1):
#     return lambda t: spdmat1
#
#
# @pytest.fixture
# def v(test_ndim):
#     return lambda t: np.ones(test_ndim)
#
#
# @pytest.fixture
# def linear_transition(G, v, S, test_ndim):
#     return pnfss.DiscreteLinearGaussian(
#         G, v, S, input_dim=test_ndim, output_dim=test_ndim
#     )
#
#
# def test_state_trans_mat(linear_transition, G):
#     received = linear_transition.state_trans_mat_fun(0.0)
#     expected = G(0.0)
#     np.testing.assert_allclose(received, expected)
#
#
# def test_shift_vec(linear_transition, v):
#     received = linear_transition.shift_vec_fun(0.0)
#     expected = v(0.0)
#     np.testing.assert_allclose(received, expected)
#
#
# def test_input_dim(linear_transition, test_ndim):
#     assert linear_transition.input_dim == test_ndim
#
# def test_output_dim(linear_transition, test_ndim):
#     assert linear_transition.output_dim == test_ndim
#
# def test_forward_rv(linear_transition, rv1):
#     pass
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class TestDiscreteLTIGaussianTransition(unittest.TestCase, NumpyAssertions):
#     def setUp(self):
#
#         self.good_dynamicsmat = np.ones((TEST_NDIM, 4 * TEST_NDIM))
#         self.good_forcevec = np.ones(TEST_NDIM)
#         self.good_diffmat = np.ones((TEST_NDIM, TEST_NDIM)) + np.eye(TEST_NDIM)
#
#     def test_init_exceptions(self):
#
#         with self.subTest("Baseline should work"):
#             pnfss.discrete_transition.DiscreteLinearGaussian(
#                 self.good_dynamicsmat, self.good_forcevec, self.good_diffmat
#             )
#
#         with self.subTest("bad dynamics"):
#             with self.assertRaises(TypeError):
#                 pnfss.discrete_transition.DiscreteLTIGaussian(
#                     self.good_forcevec, self.good_forcevec, self.good_diffmat
#                 )
#
#         with self.subTest("bad force"):
#             with self.assertRaises(TypeError):
#                 pnfss.discrete_transition.DiscreteLTIGaussian(
#                     self.good_dynamicsmat, self.good_dynamicsmat, self.good_diffmat
#                 )
#
#         with self.subTest("bad diffusion"):
#             with self.assertRaises(TypeError):
#                 pnfss.discrete_transition.DiscreteLTIGaussian(
#                     self.good_dynamicsmat, self.good_forcevec, self.good_dynamicsmat
#                 )
#
#     def test_diffmat_cholesky(self):
#         trans = pnfss.DiscreteLTIGaussian(
#             self.good_dynamicsmat, self.good_forcevec, self.good_diffmat
#         )
#
#         # Matrix square-root
#         self.assertAllClose(
#             trans.proc_noise_cov_cholesky @ trans.proc_noise_cov_cholesky.T,
#             trans.proc_noise_cov_mat,
#         )
#
#         # Lower triangular
#         self.assertAllClose(
#             trans.proc_noise_cov_cholesky, np.tril(trans.proc_noise_cov_cholesky)
#         )
#
#         # Nonnegative diagonal
#         self.assertAllClose(
#             np.diag(trans.proc_noise_cov_cholesky),
#             np.abs(np.diag(trans.proc_noise_cov_cholesky)),
#         )
