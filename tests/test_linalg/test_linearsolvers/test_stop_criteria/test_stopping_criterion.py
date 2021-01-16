"""Tests for stopping criteria of probabilistic linear solvers."""

import numpy as np
import pytest

from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.linalg.linearsolvers.stop_criteria import (
    PosteriorContraction,
    Residual,
    StoppingCriterion,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_stop_crit_returns_bool(
    stopcrit: StoppingCriterion, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether stopping criteria return a boolean value."""
    has_converged = stopcrit(
        problem=linsys_spd,
        belief=prior,
        solver_state=None,
    )
    assert isinstance(has_converged, (bool, np.bool_))


def test_solver_state_none(
    stopcrit: StoppingCriterion, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether all stopping criteria can be computed without a solver state."""
    _ = stopcrit(
        problem=linsys_spd,
        belief=prior,
        solver_state=None,
    )


@pytest.mark.parametrize(
    "stopcrit",
    [
        Residual(),
        PosteriorContraction(),
    ],
    indirect=True,
)
def test_stops_if_true_solution(
    stopcrit: StoppingCriterion,
    linsys_spd: LinearSystem,
    belief_groundtruth: LinearSystemBelief,
):
    """Test if stopping criterion returns True for the exact solution."""
    assert stopcrit(
        problem=linsys_spd,
        belief=belief_groundtruth,
    )
