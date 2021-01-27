"""Tests for the matrix vector product observation."""
import numpy as np

from probnum.linalg.solvers.data import LinearSolverAction
from probnum.linalg.solvers.observation_ops import MatVec, ObservationOperator
from probnum.problems import LinearSystem


def test_observation_is_matrix_vector_product(
    observation_op: ObservationOperator,
    action: LinearSolverAction,
    linsys_spd: LinearSystem,
):
    """Test whether the matmul observation operator returns a matrix-vector
    multiplication with the system matrix."""
    observation = MatVec()(problem=linsys_spd, action=action)
    np.testing.assert_allclose(
        observation.obsA,
        linsys_spd.A @ action.actA,
    )
    assert np.all(observation.obsb == linsys_spd.b)
