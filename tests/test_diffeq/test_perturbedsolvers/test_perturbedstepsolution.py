import numpy as np
import pytest
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq


@pytest.fixture
def perturbed_solution():
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    testsolver = diffeq.WrappedScipyRungeKutta(rk.RK45(ode.rhs, ode.t0, y0, ode.tmax))
    sol = diffeq.PerturbedStepSolver(
        testsolver,
        noise_scale=0.1,
        perturb_function=diffeq.perturb_uniform,
    )
    return sol.solve(diffeq.AdaptiveSteps(0.1, atol=1e-4, rtol=1e-4))


def test_states(perturbed_solution):
    assert isinstance(
        perturbed_solution.states, _randomvariablelist._RandomVariableList
    )


def test_call(perturbed_solution):
    """Test for continuity of the dense output.

    Small changes of the locations should come with small changes of the
    states.
    """
    np.testing.assert_allclose(
        perturbed_solution(perturbed_solution.locations[0:]).mean,
        perturbed_solution.states[0:].mean,
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        perturbed_solution(perturbed_solution.locations[0:-1] + 1e-14).mean,
        perturbed_solution(perturbed_solution.locations[0:-1]).mean,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        perturbed_solution(perturbed_solution.locations[1:] - 1e-14).mean,
        perturbed_solution(perturbed_solution.locations[1:]).mean,
        atol=1e-12,
        rtol=1e-12,
    )


def test_len(perturbed_solution):
    np.testing.assert_allclose(
        len(perturbed_solution),
        len(perturbed_solution.locations),
        atol=1e-14,
        rtol=1e-14,
    )


def test_getitem(perturbed_solution):
    np.testing.assert_allclose(
        perturbed_solution.interpolants[1](perturbed_solution.locations[1]),
        perturbed_solution[1].mean,
        atol=1e-14,
        rtol=1e-14,
    )
