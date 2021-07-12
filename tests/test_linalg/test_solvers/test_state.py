"""Tests for the state of a probabilistic linear solver."""

import numpy as np
from pytest_cases import parametrize, parametrize_with_cases

from probnum.linalg.solvers import ProbabilisticLinearSolverState

cases_states = "cases.states"


@parametrize_with_cases("state", cases=cases_states)
def test_residual(state: ProbabilisticLinearSolverState):
    """Test whether the state computes the residual correctly."""
    linsys = state.problem
    residual = linsys.A @ state.belief.x.mean - linsys.b
    np.testing.assert_allclose(residual, state.residual)


@parametrize_with_cases("state", cases=cases_states)
def test_next_step(state: ProbabilisticLinearSolverState):
    """Test whether advancing a state to the next step updates all state attributes
    correctly."""
    initial_step = state.step
    state.next_step()

    assert initial_step + 1 == state.step
    assert state.actions[state.step] is None
    assert state.observations[state.step] is None
    assert state.residuals[state.step] is None


@parametrize_with_cases("state", cases=cases_states)
@parametrize("attr_name", ["action", "observation", "residual"])
def test_current_iter_attribute(state: ProbabilisticLinearSolverState, attr_name: str):
    """Test whether the current iteration attribute if set returns the last element of
    the attribute lists."""
    assert np.all(
        getattr(state, attr_name) == getattr(state, attr_name + "s")[state.step]
    )
