"""Test cases for belief updates of probabilistic linear solvers."""

import numpy as np
import pytest

import probnum.random_variables as rvs
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_matrix_posterior_multiplication(
    n: int,
    symlin_updated_belief: LinearSystemBelief,
    linsys_spd: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether multiplication with the posteriors over A and Ainv returns a random
    variable with the correct shape."""
    matshape = (n, 5)
    mat = random_state.random(size=matshape)
    Amat = symlin_updated_belief.A @ mat
    Ainvmat = symlin_updated_belief.Ainv @ mat
    assert isinstance(Amat, rvs.Normal)
    assert Amat.shape == (n, matshape[1])

    assert isinstance(Ainvmat, rvs.Normal)
    assert Ainvmat.shape == (n, matshape[1])


def test_multiple_actions_observations_update():
    """Test whether a single update with multiple actions and observations is identical
    to multiple sequential updates."""
    pass
