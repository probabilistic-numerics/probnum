"""Tests for Runge-Kutta initialization."""
import numpy as np
import pytest

from probnum import diffeq, randvars
from tests.test_diffeq.test_odefiltsmooth.test_initialization_routines import (
    _interface_initialize_test,
)


class TestRungeKuttaInitialization(
    _interface_initialize_test.InterfaceInitializationRoutineTest
):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rk_init = (
            diffeq.odefiltsmooth.initialization_routines.RungeKuttaInitialization(
                dt=1e-1, method="RK45"
            )
        )

    def test_call(
        self, lotka_volterra, lotka_volterra_inits, lotka_volterra_testcase_order
    ):

        prior_process = self._construct_prior_process(
            order=lotka_volterra_testcase_order,
            spatialdim=lotka_volterra.dimension,
            t0=lotka_volterra.t0,
        )

        received_rv = self.rk_init(ivp=lotka_volterra, prior_process=prior_process)

        # Extract the relevant values
        expected = lotka_volterra_inits

        # The higher derivatives will have absolute difference ~8%
        # if things work out correctly
        assert isinstance(received_rv, randvars.Normal)
        np.testing.assert_allclose(received_rv.mean, expected, rtol=0.25)
        assert np.linalg.norm(received_rv.std) > 0

    def test_is_exact(self):
        assert self.rk_init.is_exact is False

    def test_requires_jax(self):
        assert self.rk_init.requires_jax is False
