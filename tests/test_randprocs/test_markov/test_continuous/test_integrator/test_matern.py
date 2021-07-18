"""Tests for Matern processes."""


import numpy as np
import pytest

from probnum import randprocs, randvars
from tests.test_randprocs.test_markov.test_continuous import test_sde
from tests.test_randprocs.test_markov.test_continuous.test_integrator import (
    test_integrator,
)


@pytest.mark.parametrize("lengthscale", [-2.0, 2.0])
@pytest.mark.parametrize("initarg", [0.0, 2.0])
@pytest.mark.parametrize("nu", [0, 1, 4])
@pytest.mark.parametrize("wiener_process_dimension", [1, 2, 3])
@pytest.mark.parametrize("use_initrv", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
def test_matern_construction(
    lengthscale, initarg, nu, wiener_process_dimension, use_initrv, diffuse
):
    if use_initrv:
        d = (nu + 1) * wiener_process_dimension
        initrv = randvars.Normal(np.arange(d), np.diag(np.arange(1, d + 1)))
    else:
        initrv = None
    if use_initrv and diffuse:
        with pytest.warns(Warning):
            randprocs.markov.continuous.integrator.MaternProcess(
                lengthscale=lengthscale,
                initarg=initarg,
                nu=nu,
                wiener_process_dimension=wiener_process_dimension,
                initrv=initrv,
                diffuse=diffuse,
            )

    else:
        matern = randprocs.markov.continuous.integrator.MaternProcess(
            lengthscale=lengthscale,
            initarg=initarg,
            nu=nu,
            wiener_process_dimension=wiener_process_dimension,
            initrv=initrv,
            diffuse=diffuse,
        )

    isinstance(matern, randprocs.markov.continuous.integrator.MaternProcess)
    isinstance(matern, randprocs.markov.MarkovProcess)
    isinstance(
        matern.transition,
        randprocs.markov.continuous.integrator.MaternTransition,
    )


class TestMaternTransition(
    test_sde.TestLTISDE, test_integrator.TestIntegratorTransition
):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        some_nu,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        self.some_nu = some_nu
        wiener_process_dimension = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = randprocs.markov.continuous.integrator.MaternTransition(
            nu=self.some_nu,
            wiener_process_dimension=wiener_process_dimension,
            lengthscale=1.2345,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: self.transition.driftmat
        self.v = lambda t: self.transition.forcevec
        self.L = lambda t: self.transition.dispmat

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    @property
    def integrator(self):
        return self.transition
