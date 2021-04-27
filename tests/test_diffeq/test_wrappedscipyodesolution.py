import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq, randvars
from probnum.diffeq import wrappedscipysolver


def case_logistic():
    y0 = np.array([0.1])
    ode = diffeq.logistic([0.0, 1.0], y0)
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax), order=4
    )
    testsolution = testsolver.solve(diffeq.AdaptiveSteps(0.1, atol=1e-2, rtol=1e-2))
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution


def case_lotkavolterra():
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax), order=4
    )
    testsolution = testsolver.solve(diffeq.AdaptiveSteps(0.1, atol=1e-12, rtol=1e-12))
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution


def case_lorenz():
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq.lorenz([0.0, 1.0], y0)
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax), order=4
    )
    testsolution = testsolver.solve(diffeq.ConstantSteps(0.1))
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution


@pytest.fixture
def solutiontest():
    ivp = diffeq.logistic([0.0, 10], np.array([1]))
    scipysolver = rk.RK45(ivp.rhs, ivp.t0, np.array([1]), ivp.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(scipysolver, order=4)
    return testsolver.solve(diffeq.ConstantSteps(0.5))


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=".")
def test_locations(testsolution, scipysolution):
    scipy_t = scipysolution.ts
    probnum_t = testsolution.locations
    np.testing.assert_allclose(scipy_t, probnum_t, atol=1e-14, rtol=1e-14)


def test_call_isscalar(solutiontest):
    call_scalar = solutiontest(0.1)
    call_array = solutiontest([0.1, 0.2, 0.3])
    assert isinstance(call_scalar, randvars.Constant)
    assert isinstance(call_array, _randomvariablelist._RandomVariableList)


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=".")
def test_states(testsolution, scipysolution):
    scipy_states = np.array(scipysolution(scipysolution.ts)).T
    probnum_states = np.array(testsolution.states.mean)
    np.testing.assert_allclose(scipy_states, probnum_states, atol=1e-14, rtol=1e-14)


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=".")
def test_call__(testsolution, scipysolution):
    scipy_call = scipysolution(scipysolution.ts)
    probnum_call = testsolution(scipysolution.ts).mean.T
    np.testing.assert_allclose(scipy_call, probnum_call, atol=1e-14, rtol=1e-14)
