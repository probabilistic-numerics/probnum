"""Tests for Taylor-mode initialization."""
import numpy as np
import pytest

from probnum import diffeq, randprocs, randvars
from probnum.problems.zoo import diffeq as diffeq_zoo
from tests.test_diffeq.test_odefiltsmooth.test_initialization_routines import (
    _interface_initialize_test,
)
from tests.test_diffeq.test_odefiltsmooth.test_initialization_routines.utils import (
    _decorators,
    _known_initial_derivatives,
)


class TestTaylorModeInitialization(
    _interface_initialize_test.InterfaceInitializationRoutineTest
):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.taylor_init = (
            diffeq.odefiltsmooth.initialization_routines.TaylorModeInitialization()
        )

    @pytest.mark.parametrize("any_order", [0, 1, 2, 3])
    @_decorators.only_if_jax_available
    def test_call(self, any_order):
        r2b_jax = diffeq_zoo.threebody_jax()

        expected = randprocs.markov.integrator.convert.convert_derivwise_to_coordwise(
            _known_initial_derivatives.THREEBODY_INITS[
                : r2b_jax.dimension * (any_order + 1)
            ],
            num_derivatives=any_order,
            wiener_process_dimension=r2b_jax.dimension,
        )

        prior_process = self._construct_prior_process(
            order=any_order, spatialdim=r2b_jax.dimension, t0=r2b_jax.t0
        )

        received_rv = self.taylor_init(ivp=r2b_jax, prior_process=prior_process)

        assert isinstance(received_rv, randvars.Normal)
        np.testing.assert_allclose(received_rv.mean, expected)
        np.testing.assert_allclose(received_rv.std, 0.0)

    def test_is_exact(self):
        assert self.taylor_init.is_exact is True

    def test_requires_jax(self):
        assert self.taylor_init.requires_jax is True
