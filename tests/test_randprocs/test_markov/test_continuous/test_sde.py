import numpy as np
import pytest

from probnum import randprocs
from tests.test_randprocs.test_markov import test_transition


class TestSDE(test_transition.InterfaceTestTransition):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1):

        self.g = lambda t, x: np.sin(x)
        self.l = lambda t, x: spdmat1
        self.dg = lambda t, x: np.cos(x)
        self.transition = randprocs.markov.continuous.SDE(
            state_dimension=test_ndim,
            wiener_process_dimension=test_ndim,
            drift_function=self.g,
            dispersion_function=self.l,
            drift_jacobian=self.dg,
        )

    # Test access to system matrices

    def test_drift(self, some_normal_rv1):
        expected = self.g(0.0, some_normal_rv1.mean)
        received = self.transition.drift_function(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_dispersionmatrix(self, some_normal_rv1):
        expected = self.l(0.0, some_normal_rv1.mean)
        received = self.transition.dispersion_function(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_drift_jacobian(self, some_normal_rv1):
        expected = self.dg(0.0, some_normal_rv1.mean)
        received = self.transition.drift_jacobian(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_rv(some_normal_rv1, 0.0, dt=0.1)

    def test_forward_realization(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_realization(some_normal_rv1.mean, 0.0, dt=0.1)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2, 0.0, dt=0.1)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(
                some_normal_rv1.mean, some_normal_rv2, 0.0, dt=0.1
            )

    def test_input_dim(self, test_ndim):
        assert self.transition.input_dim == test_ndim

    def test_output_dim(self, test_ndim):
        assert self.transition.output_dim == test_ndim

    def test_state_dimension(self, test_ndim):
        assert self.transition.state_dimension == test_ndim

    def test_wiener_process_dimension(self, test_ndim):
        assert self.transition.wiener_process_dimension == test_ndim
