"""Test cases for belief updates of probabilistic linear solvers."""

from typing import Tuple

import numpy as np

import probnum.randvars as rvs
from probnum.linalg.solvers import LinearSolverState
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_matrix_posterior_multiplication(
    n: int,
    symlin_updated_belief: Tuple[LinearSystemBelief, LinearSolverState],
    linsys_spd: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether multiplication with the posteriors over A and Ainv returns a random
    variable with the correct shape."""
    matshape = (n, 5)
    mat = random_state.random(size=matshape)
    Amat = symlin_updated_belief[0].A @ mat
    Ainvmat = symlin_updated_belief[0].Ainv @ mat
    assert isinstance(Amat, rvs.Normal)
    assert Amat.shape == (n, matshape[1])

    assert isinstance(Ainvmat, rvs.Normal)
    assert Ainvmat.shape == (n, matshape[1])


def test_multiple_actions_observations_update():
    """Test whether a single update with multiple actions and observations is identical
    to multiple sequential updates."""
    pass
