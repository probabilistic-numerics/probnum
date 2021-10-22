"""Tests for the solution-based belief update for projected right hand side
information."""

import pathlib

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from probnum import randvars
from probnum.linalg.solvers import LinearSolverState, belief_updates, beliefs

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_belief_updates = case_modules + ".belief_updates"
cases_states = case_modules + ".states"


@parametrize_with_cases(
    "belief_update", cases=cases_belief_updates, glob="*solution_based_projected_rhs*"
)
@parametrize_with_cases(
    "state",
    cases=cases_states,
    has_tag=["has_action", "has_observation", "solution_based"],
)
def test_returns_linear_system_belief(
    belief_update: belief_updates.SolutionBasedProjectedRHSBeliefUpdate,
    state: LinearSolverState,
):
    belief = belief_update(solver_state=state)
    assert isinstance(belief, beliefs.LinearSystemBelief)


def test_negative_noise_variance_raises_error():
    with pytest.raises(ValueError):
        belief_updates.SolutionBasedProjectedRHSBeliefUpdate(noise_var=-1.0)


@parametrize_with_cases(
    "belief_update", cases=cases_belief_updates, glob="*solution_based_projected_rhs*"
)
@parametrize_with_cases(
    "state",
    cases=cases_states,
    has_tag=["has_action", "has_observation", "solution_based"],
)
def test_beliefs_against_naive_implementation(
    belief_update: belief_updates.SolutionBasedProjectedRHSBeliefUpdate,
    state: LinearSolverState,
):
    """Compare the updated belief to a naive implementation."""
    # Belief update
    updated_belief = belief_update(solver_state=state)

    # Naive implementation
    belief = state.belief
    action = state.action
    observ = state.observation
    noise_var = belief_update._noise_var

    A_action = state.problem.A @ action
    gram = A_action.T @ belief.x.cov @ A_action + noise_var
    gram_pinv = 1.0 / gram if gram > 0.0 else 0.0

    x = randvars.Normal(
        mean=belief.x.mean
        + belief.x.cov @ A_action * (observ - A_action.T @ belief.x.mean) * gram_pinv,
        cov=belief.x.cov
        - (belief.x.cov @ A_action) @ (belief.x.cov @ A_action).T * gram_pinv,
    )
    Ainv = (
        belief.Ainv
        + (belief.x.cov @ A_action) @ (belief.x.cov @ A_action).T * gram_pinv
    )

    naive_belief = beliefs.LinearSystemBelief(x=x, Ainv=Ainv)

    # Compare means and covariances
    np.testing.assert_allclose(
        updated_belief.x.mean,
        naive_belief.x.mean,
        err_msg="Mean of solution belief does not match naive implementation.",
        atol=1e-12,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        updated_belief.x.cov,
        naive_belief.x.cov,
        err_msg="Covariance of solution belief does not match naive implementation.",
        atol=1e-12,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        updated_belief.Ainv.mean.todense(),
        naive_belief.Ainv.mean.todense(),
        err_msg="Belief about the inverse does not match naive implementation.",
        atol=1e-12,
        rtol=1e-12,
    )
