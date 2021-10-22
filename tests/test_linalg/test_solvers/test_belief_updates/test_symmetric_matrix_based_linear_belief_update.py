"""Tests for the symmetric matrix-based belief update for linear information."""

import pathlib

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from probnum import linops, randvars
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


@parametrize_with_cases(
    "belief_update", cases=cases_belief_updates, glob="symmetric_matrix_based_linear*"
)
@parametrize_with_cases(
    "state",
    cases=cases_states,
    has_tag=["has_action", "has_observation", "symmetric_matrix_based"],
)
def test_against_naive_implementation(
    belief_update: belief_updates.MatrixBasedLinearBeliefUpdate,
    state: LinearSolverState,
):
    """Compare the updated belief to a naive implementation."""

    def dense_matrix_based_update(
        matrix: randvars.Normal, action: np.ndarray, observ: np.ndarray
    ):
        pred = matrix.mean @ action
        resid = observ - pred
        covfactor_Ms = matrix.cov.A @ action
        gram = action.T @ covfactor_Ms
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0
        gain = covfactor_Ms * gram_pinv
        covfactor_update = gain @ covfactor_Ms.T
        resid_gain = np.outer(resid, gain)

        return randvars.Normal(
            mean=matrix.mean
            + resid_gain
            + resid_gain.T
            - np.outer(gain, action.T @ resid_gain),
            cov=linops.SymmetricKronecker(A=matrix.cov.A - covfactor_update),
        )

    updated_belief = belief_update(solver_state=state)
    A_naive = dense_matrix_based_update(
        matrix=state.belief.A, action=state.action, observ=state.observation
    )
    Ainv_naive = dense_matrix_based_update(
        matrix=state.belief.Ainv, action=state.observation, observ=state.action
    )

    # System matrix
    np.testing.assert_allclose(
        updated_belief.A.mean.todense(),
        A_naive.mean.todense(),
        err_msg="Mean of system matrix estimate does not match naive implementation.",
    )
    np.testing.assert_allclose(
        updated_belief.A.cov.todense(),
        A_naive.cov.todense(),
        err_msg="Covariance of system matrix estimate does not match naive implementation.",
    )

    # Inverse
    np.testing.assert_allclose(
        updated_belief.Ainv.mean.todense(),
        Ainv_naive.mean.todense(),
        err_msg="Mean of matrix inverse estimate does not match naive implementation.",
    )
    np.testing.assert_allclose(
        updated_belief.Ainv.cov.todense(),
        Ainv_naive.cov.todense(),
        err_msg="Covariance of matrix inverse estimate does not match naive implementation.",
    )
