"""Tests for observation operators of probabilistic linear solvers."""

import numpy as np

from probnum.linalg.linearsolvers.observation_ops import (
    MatVecObservation,
    ObservationOperator,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_observation_is_vector_or_float(
    observation_op: ObservationOperator, action: np.ndarray, linsys_spd: LinearSystem
):
    """Test whether observation operators return a vector."""
    observation, _ = observation_op(
        problem=linsys_spd,
        action=action,
        solver_state=None,
    )
    assert (
        isinstance(observation, (np.ndarray, float, np.float_)),
        f"Observation {observation} returned by"
        f" {observation_op.__class__.__name__} "
        f"is not an np.ndarray.",
    )


# Matrix-vector product observations


def test_observation_is_matrix_vector_product(
    observation_op: ObservationOperator, action: np.ndarray, linsys_spd: LinearSystem
):
    """Test whether the matmul observation operator returns a matrix-vector
    multiplication with the system matrix."""
    np.testing.assert_allclose(
        MatVecObservation()(problem=linsys_spd, action=action)[0],
        linsys_spd.A @ action,
    )
