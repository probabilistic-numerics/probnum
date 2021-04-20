import numpy as np
import pytest
from pn_ode_benchmarks import scipy_solver
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq
from probnum.diffeq.perturbedsolvers import perturbedstepsolution, perturbedstepsolver


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
    return perturbedstepsolver.PerturbedStepSolver(scipysolver, sigma, "uniform")


@pytest.fixture
def steprule(stepsize):
    return diffeq.ConstantSteps(stepsize)


@pytest.fixture
def solutiontest(testsolver45, steprule):
    return testsolver45.solve(steprule)


@pytest.fixture
def solutionnoisy(noisysolver45, steprule):
    return noisysolver45.solve(steprule)


def test_t(solutionnoisy, start, stop, stepsize):
    noisy_times = solutionnoisy.t
    original_t = np.arange(start, stop + stepsize, stepsize)
    np.testing.assert_allclose(
        noisy_times[0:5], original_t[0:5], atol=1e-13, rtol=1e-13
    )


def test_states(solutionnoisy):
    assert isinstance(solutionnoisy.states, _randomvariablelist._RandomVariableList)


def test_call(solutionnoisy):
    np.testing.assert_allclose(
        solutionnoisy(solutionnoisy.t).mean,
        solutionnoisy.states.mean,
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        solutionnoisy(solutionnoisy.t + 1e-14).mean,
        solutionnoisy(solutionnoisy.t).mean,
        atol=1e-10,
        rtol=1e-10,
    )


def test_len(solutionnoisy):
    np.testing.assert_allclose(
        len(solutionnoisy), len(solutionnoisy.t), atol=1e-14, rtol=1e-14
    )


def test_getitem(solutionnoisy):
    np.testing.assert_allclose(
        solutionnoisy.interpolants[1](solutionnoisy.t[1]),
        solutionnoisy[1].mean,
        atol=1e-14,
        rtol=1e-14,
    )


def test_sample(solutionnoisy):
    np.testing.assert_string_equal(solutionnoisy.sample(5), "Sampling not possible")
