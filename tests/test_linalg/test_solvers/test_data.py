"""Tests for the objects storing the data collected by a linear solver."""

import numpy as np

from probnum.linalg.solvers import LinearSolverData


def test_data_arrays(solver_data: LinearSolverData):
    """Test whether the linear solver data is correctly converted to arrays."""
    assert np.all(
        solver_data.actions_arr.actA
        == np.hstack([action.actA for action in solver_data.actions])
    )
    assert np.all(
        solver_data.observations_arr.obsA
        == np.hstack([obs.obsA for obs in solver_data.observations])
    )


def test_data_array_shape(solver_data: LinearSolverData):
    """Test whether arrays of linear solver data have the correct shape."""
    assert solver_data.actions_arr.actA.shape == (
        solver_data.actions[0].actA.shape[0],
        len(solver_data.actions),
    )

    assert solver_data.observations_arr.obsA.shape == (
        solver_data.observations[0].obsA.shape[0],
        len(solver_data.observations),
    )


def test_from_arrays_returns_same_arrays(solver_data: LinearSolverData):
    """Test whether creating linear solver data from arrays returns the same arrays."""

    new_solver_data = LinearSolverData.from_arrays(
        actions_arr=(
            solver_data.actions_arr.actA,
            solver_data.actions_arr.actb,
        ),
        observations_arr=(
            solver_data.observations_arr.obsA,
            solver_data.observations_arr.obsb,
        ),
    )

    np.testing.assert_equal(
        solver_data.actions_arr.actA, new_solver_data.actions_arr.actA
    )
    np.testing.assert_equal(
        solver_data.actions_arr.actb, new_solver_data.actions_arr.actb
    )
    np.testing.assert_equal(
        solver_data.observations_arr.obsA, new_solver_data.observations_arr.obsA
    )
    np.testing.assert_equal(
        solver_data.observations_arr.obsb, new_solver_data.observations_arr.obsb
    )
