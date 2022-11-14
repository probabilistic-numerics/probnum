import numpy as np

from probnum import randprocs, randvars

import pytest
from tests.probnum.randprocs.markov.discrete import test_linear_gaussian


class TestLTIGaussian(test_linear_gaussian.TestLinearGaussian):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        state_dim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.transition_matrix = spdmat1
        self.noise = randvars.Normal(mean=np.arange(state_dim), cov=spdmat2)

        self.transition = randprocs.markov.discrete.LTIGaussian(
            transition_matrix=self.transition_matrix,
            noise=self.noise,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        # Compatibility with superclass' test
        self.transition_matrix_fun = lambda t: self.transition_matrix
        self.noise_fun = lambda t: self.noise
        self.transition_fun = lambda t, x: self.transition_matrix @ x
        self.transition_fun_jacobian = lambda t, x: self.transition_matrix

    # Test access to system matrices

    def test_transition_matrix(self):
        received = self.transition.transition_matrix
        expected = self.transition_matrix
        np.testing.assert_allclose(received, expected)

    def test_noise(self):
        received = self.transition.noise
        expected = self.noise
        np.testing.assert_allclose(received.mean, expected.mean)
        np.testing.assert_allclose(received.cov, expected.cov)
