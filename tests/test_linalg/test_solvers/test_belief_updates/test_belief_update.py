"""Test cases for belief updates of probabilistic linear solvers."""

import numpy as np
import pytest

import probnum.random_variables as rvs
from probnum.linalg.solvers.belief_updates import LinearSolverBeliefUpdate
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"

pytestmark = pytest.mark.usefixtures("linobs_belief_update")


def test_matrix_posterior_multiplication(
    n: int,
    linobs_belief_update: LinearSolverBeliefUpdate,
    linsys_spd: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether multiplication with the posteriors over A and Ainv returns a random
    variable with the correct shape."""
    x, Ainv, A, b, solver_state = linobs_belief_update()
    matshape = (n, 5)
    mat = random_state.random(size=matshape)
    Amat = A @ mat
    Ainvmat = Ainv @ mat
    assert isinstance(Amat, rvs.Normal)
    assert Amat.shape == (n, matshape[1])

    assert isinstance(Ainvmat, rvs.Normal)
    assert Ainvmat.shape == (n, matshape[1])


# def test_multiple_actions_observations_update(self):
#     """Test whether a single update with multiple actions and observations is
#     identical to multiple sequential updates."""
#     n_iterations = 5
#     actions = self.rng.normal(size=(self.linsys.shape[0], n_iterations))
#     observations = self.linsys.A @ actions
#
#     for belief_update in self.belief_updates:
#         with self.subTest():
#             belief_bulk, _ = belief_update(
#                 problem=self.linsys,
#                 belief=self.prior,
#                 action=actions,
#                 observation=observations,
#                 solver_state=self.solver_state,
#             )
#
#             belief_iter = self.prior
#             for i in range(n_iterations):
#                 belief_iter, _ = belief_update(
#                     problem=self.linsys,
#                     belief=belief_iter,
#                     action=actions[:, i][:, None],
#                     observation=observations[:, i][:, None],
#                     solver_state=self.solver_state,
#                 )
#
#             self.assertAllClose(
#                 belief_bulk.A.mean.todense(), belief_iter.A.mean.todense()
#             )
#             self.assertAllClose(
#                 belief_bulk.Ainv.mean.todense(), belief_iter.Ainv.mean.todense()
#             )
#             self.assertAllClose(belief_bulk.x.mean, belief_iter.x.mean)
#             self.assertAllClose(belief_bulk.b.mean, belief_iter.b.mean)
