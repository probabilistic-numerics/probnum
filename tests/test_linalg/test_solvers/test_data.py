"""Tests for the objects storing the data collected by a linear solver."""

import numpy as np

from probnum.linalg.solvers import LinearSolverData


def test_data_arrays(solver_data: LinearSolverData):
    """Test whether the linear solver data is correctly converted to arrays."""
    assert np.all(
        solver_data.actions_arr.A
        == np.hstack([action.A for action in solver_data.actions])
    )
    assert np.all(
        solver_data.observations_arr.A
        == np.hstack([obs.A for obs in solver_data.observations])
    )
