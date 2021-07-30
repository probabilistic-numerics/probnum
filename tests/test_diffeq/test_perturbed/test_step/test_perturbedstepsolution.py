import numpy as np
import pytest
from scipy.integrate._ivp import rk

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import _randomvariablelist, diffeq


@pytest.fixture
def steprule():
    return diffeq.stepsize.AdaptiveSteps(0.1, atol=1e-4, rtol=1e-4)


@pytest.fixture
def perturbed_solution(steprule):
    y0 = np.array([0.1, 0.1])
    ode = diffeq_zoo.lotkavolterra(t0=0.0, tmax=1.0, y0=y0)
    rng = np.random.default_rng(seed=1)
    testsolver = diffeq.perturbed.scipy_wrapper.WrappedScipyRungeKutta(
        rk.RK45, steprule=steprule
    )
    sol = diffeq.perturbed.step.PerturbedStepSolver(
        rng=rng,
        solver=testsolver,
        noise_scale=0.1,
        perturb_function=diffeq.perturbed.step.perturb_uniform,
    )
    return sol.solve(ode)


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
