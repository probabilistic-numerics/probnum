"""Tests for the state of a probabilistic linear solver."""

import numpy as np
import pytest

from probnum.linalg.solvers import (
    LinearSolverData,
    LinearSolverInfo,
    LinearSolverMiscQuantities,
    LinearSolverState,
)
from probnum.linalg.solvers.beliefs import LinearSystemBelief


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


class TestLinearSolverData:
    """Tests for the objects storing the data collected by a linear solver."""

    def test_data_arrays(self, solver_data: LinearSolverData):
        """Test whether the linear solver data is correctly converted to arrays."""
        assert np.all(solver_data.actions_arr == np.hstack(solver_data.actions))
        assert np.all(
            solver_data.observations_arr == np.hstack(solver_data.observations)
        )


class TestLinearSolverMiscQuantities:
    """Tests for the miscellaneous (cached) quantities stored for efficiency."""

    def test_from_new_data_clears_cached_residual(
        self,
        action: np.ndarray,
        matvec_observation: np.ndarray,
        solver_misc_quantities: LinearSolverMiscQuantities,
    ):
        """Test whether adding new data clears cached property 'residual'."""
        solver_misc_quantities_new = LinearSolverMiscQuantities.from_new_data(
            action=action, observation=matvec_observation, prev=solver_misc_quantities
        )

        with pytest.raises(KeyError):
            _ = solver_misc_quantities_new.__dict__["residual"]


class TestLinearSolverState:
    """Tests for the linear solver state."""

    def test_from_updated_belief_preserves_cached_residual(
        self, symm_belief: LinearSystemBelief, solver_state_init: LinearSolverState
    ):
        """Tests whether updating a solver state with new data preserves the cached
        residual."""
        # Cache residual
        _ = solver_state_init.misc.residual

        # Updated state
        new_state = LinearSolverState.from_updated_belief(
            updated_belief=symm_belief, prev_state=solver_state_init
        )

        # Check for existence of cached residual and compare
        assert "residual" in new_state.misc.__dict__.keys()
        assert np.all(solver_state_init.misc.residual == new_state.misc.residual)
