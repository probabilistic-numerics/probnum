"""Tests for observation operators of probabilistic linear solvers."""

import numpy as np

from probnum.linalg.solvers.observation_ops import ObservationOp
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_observation_array(
    observation_op: ObservationOp, action: np.ndarray, linsys_spd: LinearSystem
):
    """Test whether observation operators return a vector."""
    observation = observation_op(
        problem=linsys_spd,
        action=action,
        solver_state=None,
    )
    assert isinstance(observation.obsA, np.ndarray)
    assert isinstance(observation.obsb, np.ndarray)
