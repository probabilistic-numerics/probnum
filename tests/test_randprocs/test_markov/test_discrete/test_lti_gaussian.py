import numpy as np
import pytest

from probnum import randprocs, randvars
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

        self.transition_matrix = spdmat1
        self.process_noise = randvars.Normal(mean=np.arange(test_ndim), cov=spdmat2)
        self.transition = randprocs.markov.discrete.LTIGaussian(
            transition_matrix=self.transition_matrix,
            process_noise=self.process_noise,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        # Compatibility with superclass' test
        self.G = lambda t: self.transition_matrix
        self.process_noise_fun = lambda t: self.process_noise
        self.g = lambda t, x: self.G(t) @ x
        self.dg = lambda t, x: self.G(t)

    # Test access to system matrices

    def test_transition_matrix(self):
        received = self.transition.transition_matrix
        expected = self.transition_matrix
        np.testing.assert_allclose(received, expected)

    def test_process_noise(self):
        received = self.transition.process_noise
        expected = self.process_noise
        np.testing.assert_allclose(received.mean, expected.mean)
        np.testing.assert_allclose(received.cov, expected.cov)
