"""Tests for Runge-Kutta initialization."""
import numpy as np
import pytest

from probnum import diffeq, randprocs, randvars, statespace
from probnum.problems.zoo import diffeq as diffeq_zoo
from tests.test_diffeq.test_odefiltsmooth.test_initialize import (
    _interface_initialize_test,
)
from tests.test_diffeq.test_odefiltsmooth.test_initialize.utils import (
    _decorators,
    _known_initial_derivatives,
)


class TestTaylorModeInitialization(
    _interface_initialize_test.InterfaceInitializationRoutineTest
):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.taylor_init = diffeq.odefiltsmooth.initialize.TaylorModeInitialization()

    @pytest.mark.parametrize("any_order", [0, 1, 2])
    @_decorators.only_if_jax_available
    def test_call(self, any_order):
        r2b_jax = diffeq_zoo.threebody_jax()
        ode_dim = 4
        expected = statespace.Integrator._convert_derivwise_to_coordwise(
            _known_initial_derivatives.THREEBODY_INITS[: ode_dim * (any_order + 1)],
            ordint=any_order,
            spatialdim=ode_dim,
        )

        prior = statespace.IBM(
            ordint=any_order,
            spatialdim=ode_dim,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )

        initrv = randvars.Normal(np.zeros(prior.dimension), np.eye(prior.dimension))
        prior_process = randprocs.MarkovProcess(
            transition=prior, initrv=initrv, initarg=r2b_jax.t0
        )

        taylor_init = diffeq.odefiltsmooth.initialize.TaylorModeInitialization()
        received_rv = taylor_init(ivp=r2b_jax, prior_process=prior_process)

        np.testing.assert_allclose(received_rv.mean, expected)
        np.testing.assert_allclose(received_rv.std, 0.0)

    def test_is_exact(self):
        assert self.taylor_init.is_exact is True

    def test_requires_jax(self):
        assert self.taylor_init.requires_jax is True
