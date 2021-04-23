import numpy as np
import pytest
from pn_ode_benchmarks import scipy_solver
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq
from probnum.diffeq.perturbedsolvers import (
    _perturbation_functions,
    perturbedstepsolution,
    perturbedstepsolver,
)


@pytest.fixture
def y0():
    return np.array([0.1])


@pytest.fixture
def start():
    return 0.0


@pytest.fixture
def stop():
    return 10.0


@pytest.fixture
def ivp(start, stop, y0):
    return diffeq.logistic([start, stop], y0)


@pytest.fixture
def sigma():
    return 1


@pytest.fixture
def stepsize():
    return 0.1


@pytest.fixture
def testsolver45(ivp, y0):
    testsolver = rk.RK45(ivp.rhs, ivp.t0, y0, ivp.tmax)
    return scipy_solver.ScipyRungeKutta(testsolver, order=4)


@pytest.fixture
def noisysolver45(ivp, y0, sigma):
    testsolver = rk.RK45(ivp.rhs, ivp.t0, y0, ivp.tmax)
    scipysolver = scipy_solver.ScipyRungeKutta(testsolver, order=4)
    return perturbedstepsolver.PerturbedStepSolver(
        scipysolver, sigma, perturb_function=_perturbation_functions.perturb_uniform
    )


@pytest.fixture
def steprule(stepsize):
    return diffeq.AdaptiveSteps(firststep=0.001, atol=10 ** -12, rtol=10 ** -12)


@pytest.fixture
def solutiontest(testsolver45, steprule):
    return testsolver45.solve(steprule)


@pytest.fixture
def solutionnoisy(noisysolver45, steprule):
    return noisysolver45.solve(steprule)


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
        ([0.0, 1.0, 2.0], 0.5, 0.0),
        ([0.0, 1.0, 2.0], 1.1, 1.0),
        ([0.0, 1.0, 2.0], 2.0, 1.0),
        ([0.0, 1.0, 2.0], 1.9, 1.0),
    ],
)
def test_get_interpolant(array, element, pos):
    closest = perturbedstepsolution.get_interpolant(array, element)
    assert closest == pos
