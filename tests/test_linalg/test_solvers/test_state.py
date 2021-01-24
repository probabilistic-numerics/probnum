"""Tests for the state of a probabilistic linear solver."""

import numpy as np
import pytest

from probnum.linalg.solvers import (
    LinearSolverCache,
    LinearSolverInfo,
    LinearSolverState,
)
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.data import LinearSolverAction, LinearSolverObservation


class TestLinearSolverInfo:
    """Tests for the convergence information of a linear solver."""

    def test_has_converged_stopping_criterion_consistency(
        self, solver_info: LinearSolverInfo
    ):
        """Test whether the boolean convergence flag and stopping criterion are
        consistent with each other."""
        if solver_info.has_converged:
            assert solver_info.stopping_criterion is not None
        else:
            assert solver_info.stopping_criterion is None


class TestLinearSolverCache:
    """Tests for the miscellaneous (cached) quantities stored for efficiency."""

    def test_from_new_data_clears_cached_residual(
        self,
        action: LinearSolverAction,
        matvec_observation: LinearSolverObservation,
        solver_cache: LinearSolverCache,
    ):
        """Test whether adding new data clears cached property 'residual'."""
        solver_cache_new = LinearSolverCache.from_new_data(
            action=action,
            observation=matvec_observation,
            prev_cache=solver_cache,
        )

        with pytest.raises(KeyError):
            _ = solver_cache_new.__dict__["residual"]


class TestLinearSolverState:
    """Tests for the linear solver state."""

    def test_from_updated_belief_preserves_cached_residual(
        self, symm_belief: LinearSystemBelief, solver_state_init: LinearSolverState
    ):
        """Tests whether updating a solver state with new data preserves the cached
        residual."""
        # Cache residual
        _ = solver_state_init.cache.residual

        # Updated state
        new_state = LinearSolverState.from_updated_belief(
            updated_belief=symm_belief, prev_state=solver_state_init
        )

        # Check for existence of cached residual and compare
        assert "residual" in new_state.cache.__dict__.keys()
        assert np.all(solver_state_init.cache.residual == new_state.cache.residual)
