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

        self.G_const = spdmat1
        self.process_noise = randvars.Normal(mean=np.arange(test_ndim), cov=spdmat2)
        self.transition = randprocs.markov.discrete.LTIGaussian(
            state_trans_mat=self.G_const,
            process_noise=self.process_noise,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        # Compatibility with superclass' test
        self.G = lambda t: self.G_const
        self.process_noise_fun = lambda t: self.process_noise
        # self.S = lambda t: self.process_noise.cov
        # self.v = lambda t: self.process_noise.mean
        self.g = lambda t, x: self.G(t) @ x + self.process_noise_fun(t).mean
        self.dg = lambda t, x: self.G(t)

    # Test access to system matrices

    def test_state_transition_mat(self):
        received = self.transition.state_trans_mat
        expected = self.G_const
        np.testing.assert_allclose(received, expected)

    def test_process_noise(self):
        received = self.transition.process_noise
        expected = self.process_noise
        np.testing.assert_allclose(received.mean, expected.mean)
        np.testing.assert_allclose(received.cov, expected.cov)
