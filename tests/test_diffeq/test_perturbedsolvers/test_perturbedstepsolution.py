import numpy as np
import pytest
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq
from probnum.diffeq import wrappedscipysolver
from probnum.diffeq.perturbedsolvers import (
    _perturbation_functions,
    perturbedstepsolution,
    perturbedstepsolver,
)


@pytest.fixture
def solutionnoisy():
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    )
    sol = perturbedstepsolver.PerturbedStepSolver(
        testsolver,
        noise_scale=0.1,
        perturb_function=_perturbation_functions.perturb_uniform,
        # random_state=123,
    )
    return sol.solve(diffeq.AdaptiveSteps(0.1, atol=1e-14, rtol=1e-14))


def test_states(solutionnoisy):
    assert isinstance(solutionnoisy.states, _randomvariablelist._RandomVariableList)


def test_call(solutionnoisy):
    np.testing.assert_allclose(
        solutionnoisy(solutionnoisy.locations[0:]).mean,
        solutionnoisy.states[0:].mean,
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        solutionnoisy(solutionnoisy.locations[0:-1] + 1e-14).mean,
        solutionnoisy(solutionnoisy.locations[0:-1]).mean,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        solutionnoisy(solutionnoisy.locations[1:] - 1e-14).mean,
        solutionnoisy(solutionnoisy.locations[1:]).mean,
        atol=1e-12,
        rtol=1e-12,
    )


def test_len(solutionnoisy):
    np.testing.assert_allclose(
        len(solutionnoisy), len(solutionnoisy.locations), atol=1e-14, rtol=1e-14
    )


def test_getitem(solutionnoisy):
    np.testing.assert_allclose(
        solutionnoisy.interpolants[1](solutionnoisy.locations[1]),
        solutionnoisy[1].mean,
        atol=1e-14,
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    "array, element, pos",
    [
        ([0.0, 1.0, 2.0], 0.0, 0.0),
        ([0.0, 1.0, 2.0], -1.0, 0.0),
        ([0.0, 1.0, 2.0], 0.5, 0.0),
        ([0.0, 1.0, 2.0], 1.1, 1.0),
        ([0.0, 1.0, 2.0], 2.0, 1.0),
        ([0.0, 1.0, 2.0], 1.9, 1.0),
    ],
)
def test_get_interpolant(array, element, pos):
    closest = perturbedstepsolution.get_interpolant(array, element)
    assert closest == pos
