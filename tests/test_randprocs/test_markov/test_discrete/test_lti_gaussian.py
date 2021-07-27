import numpy as np
import pytest

from probnum import randprocs
from tests.test_randprocs.test_markov.test_discrete import test_linear_gaussian


class TestLTIGaussian(test_linear_gaussian.TestLinearGaussian):

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
        self.S_const = spdmat2
        self.v_const = np.arange(test_ndim)
        self.transition = randprocs.markov.discrete.LTIGaussian(
            self.G_const,
            self.v_const,
            self.S_const,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        # Compatibility with superclass' test
        self.G = lambda t: self.G_const
        self.S = lambda t: self.S_const
        self.v = lambda t: self.v_const
        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    # Test access to system matrices

    def test_state_transition_mat(self):
        received = self.transition.state_trans_mat
        expected = self.G_const
        np.testing.assert_allclose(received, expected)

    def test_shift_vec(self):
        received = self.transition.shift_vec
        expected = self.v_const
        np.testing.assert_allclose(received, expected)

    def test_process_noise_cov_mat(self):
        received = self.transition.proc_noise_cov_mat
        expected = self.S_const
        np.testing.assert_allclose(received, expected)

    def test_process_noise_cov_cholesky(self):
        received = self.transition.proc_noise_cov_cholesky
        expected = np.linalg.cholesky(self.S_const)
        np.testing.assert_allclose(received, expected)
