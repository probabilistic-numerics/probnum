"""Tests for the matrix vector product observation."""
import numpy as np
from problems import LinearSystem

from probnum.linalg.linearsolvers.observation_ops import (
    MatVecObservation,
    ObservationOperator,
)


def test_observation_is_matrix_vector_product(
    observation_op: ObservationOperator, action: np.ndarray, linsys_spd: LinearSystem
):
    """Test whether the matmul observation operator returns a matrix-vector
    multiplication with the system matrix."""
    np.testing.assert_allclose(
        MatVecObservation()(problem=linsys_spd, action=action)[0],
        linsys_spd.A @ action,
    )
