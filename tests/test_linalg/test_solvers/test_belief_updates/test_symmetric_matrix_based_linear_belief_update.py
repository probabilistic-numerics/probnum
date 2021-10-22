"""Tests for the symmetric matrix-based belief update for linear information."""

import pathlib

import pytest
from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import LinearSolverState, belief_updates, beliefs

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_belief_updates = case_modules + ".belief_updates"
cases_states = case_modules + ".states"


@parametrize_with_cases(
    "belief_update",
    cases=cases_belief_updates,
    glob="symmetric_matrix_based_linear*",
)
@parametrize_with_cases(
    "state",
    cases=cases_states,
    has_tag=["has_action", "has_observation", "symmetric_matrix_based"],
)
def test_returns_linear_system_belief(
    belief_update: belief_updates.SymmetricMatrixBasedLinearBeliefUpdate,
    state: LinearSolverState,
):
    belief = belief_update(solver_state=state)
    assert isinstance(belief, beliefs.LinearSystemBelief)


@parametrize_with_cases(
    "belief_update", cases=cases_belief_updates, glob="symmetric_matrix_based_linear*"
)
@parametrize_with_cases(
    "state",
    cases=cases_states,
    has_tag=["has_action", "has_observation", "matrix_based"],
)
def test_raises_error_for_non_symmetric_Kronecker_structured_covariances(
    belief_update: belief_updates.SymmetricMatrixBasedLinearBeliefUpdate,
    state: LinearSolverState,
):
    with pytest.raises(ValueError):
        belief_update(solver_state=state)


def test_against_naive_implementation():
    """Compare the updated belief to a naive implementation."""
    pass  # TODO
