"""Tests for Runge-Kutta initialization."""
import numpy as np
import pytest

from probnum import diffeq, randprocs, randvars, statespace
from tests.test_diffeq.test_odefiltsmooth.test_initialize import (
    _interface_initialize_test,
)


class TestRungeKuttaInitialization(
    _interface_initialize_test.InterfaceInitializationRoutineTest
):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rk_init = diffeq.odefiltsmooth.initialize.RungeKuttaInitialization(
            dt=1e-1, method="RK45"
        )

    def test_call(self, lotka_volterra, lotka_volterra_inits, lotka_volterra_order):
        ode_dim = len(lotka_volterra.y0)
        prior = statespace.IBM(
            ordint=lotka_volterra_order,
            spatialdim=ode_dim,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        initrv = randvars.Normal(
            np.zeros(prior.dimension),
            np.eye(prior.dimension),
            cov_cholesky=np.eye(prior.dimension),
        )
        prior_process = randprocs.MarkovProcess(
            transition=prior, initrv=initrv, initarg=lotka_volterra.t0
        )

        received_rv = self.rk_init(ivp=lotka_volterra, prior_process=prior_process)

        # Extract the relevant values
        expected = lotka_volterra_inits

        # The higher derivatives will have absolute difference ~8%
        # if things work out correctly
        np.testing.assert_allclose(received_rv.mean, expected, rtol=0.25)
        assert np.linalg.norm(received_rv.std) > 0

    def test_is_exact(self):
        assert self.rk_init.is_exact is False

    def test_requires_jax(self):
        assert self.rk_init.requires_jax is False
