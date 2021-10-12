import numpy as np
import pytest_cases

from probnum import _randomvariablelist, randvars


@pytest_cases.fixture
@pytest_cases.parametrize_with_cases(
    "testsolver, scipysolver, ode", cases=".test_wrapped_scipy_cases"
)
def solution_case(testsolver, scipysolver, ode):
    testsolution = testsolver.solve(ode)
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution, ode


def test_locations(solution_case):
    testsolution, scipysolution, ode = solution_case
    scipy_t = scipysolution.ts
    probnum_t = testsolution.locations
    np.testing.assert_allclose(scipy_t, probnum_t, atol=1e-13, rtol=1e-13)


def test_call_isscalar(solution_case):
    testsolution, scipysolution, ode = solution_case
    t = 0.1
    call_scalar = testsolution(t)
    call_array = testsolution([0.1, 0.2, 0.3])
    assert np.isscalar(t)
    assert isinstance(call_scalar, randvars.Constant)
    assert isinstance(call_array, _randomvariablelist._RandomVariableList)


def test_states(solution_case):
    testsolution, scipysolution, ode = solution_case
    scipy_states = np.array(scipysolution(scipysolution.ts)).T
    probnum_states = np.array(testsolution.states.mean)
    np.testing.assert_allclose(scipy_states, probnum_states, atol=1e-13, rtol=1e-13)


def test_call(solution_case):
    testsolution, scipysolution, ode = solution_case
    scipy_call = scipysolution(scipysolution.ts)
    probnum_call = testsolution(scipysolution.ts).mean.T
    np.testing.assert_allclose(scipy_call, probnum_call, atol=1e-13, rtol=1e-13)
