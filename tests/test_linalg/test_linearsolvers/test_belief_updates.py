"""Test cases for belief updates of probabilistic linear solvers."""
from typing import Optional

import numpy as np

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState, LinearSystemBelief
from probnum.linalg.linearsolvers.belief_updates import (
    BeliefUpdate,
    LinearSymmetricGaussian,
)
from probnum.problems import LinearSystem
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class BeliefUpdateTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for belief updates of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver belief updates."""

        # Belief updates
        def custom_belief_update(
            problem: LinearSystem,
            belief: LinearSystemBelief,
            action: np.ndarray,
            observation: np.ndarray,
            solver_state: Optional[LinearSolverState] = None,
        ):
            # Richardson iteration
            omega = 0.01
            belief.x += omega * (problem.b - problem.A @ belief.x.mean)

            return belief, solver_state

        self.custom_belief_update = BeliefUpdate(belief_update=custom_belief_update)
        self.linear_symmetric_gaussian = LinearSymmetricGaussian()
        self.belief_updates = [
            self.custom_belief_update,
            self.linear_symmetric_gaussian,
        ]

    def test_return_argument_types(self):
        """Test whether a belief update returns a linear system belief and solver
        state."""

        for belief_update in self.belief_updates:
            with self.subTest():
                belief, solver_state = belief_update(
                    problem=self.linsys,
                    belief=self.prior,
                    action=self.action,
                    observation=self.observation,
                    solver_state=self.solver_state,
                )
                self.assertIsInstance(belief, LinearSystemBelief)
                self.assertIsInstance(solver_state, LinearSolverState)

    def test_matrix_posterior_multiplication(self):
        """Test whether multiplication with the posteriors over A and Ainv returns a
        random variable with the correct shape."""
        for belief_update in self.belief_updates:
            with self.subTest():
                belief, solver_state = belief_update(
                    problem=self.linsys,
                    belief=self.prior,
                    action=self.action,
                    observation=self.observation,
                    solver_state=self.solver_state,
                )
                matshape = (self.linsys.A.shape[1], 5)
                mat = self.rng.random(size=matshape)
                Amat = belief.A @ mat
                Ainvmat = belief.Ainv @ mat
                self.assertIsInstance(Amat, rvs.Normal)
                self.assertEqual(Amat.shape, (self.linsys.A.shape[0], matshape[1]))

                self.assertIsInstance(Ainvmat, rvs.Normal)
                self.assertEqual(Ainvmat.shape, (self.linsys.A.shape[0], matshape[1]))

    #
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


class LinearSymmetricGaussianTestCase(BeliefUpdateTestCase):
    """Test case for the linear symmetric Gaussian belief update."""

    def setUp(self) -> None:
        """Test resources for the linear Gaussian belief update."""
        self.belief_updates = [LinearSymmetricGaussian()]

    def test_symmetric_posterior_params(self):
        """Test whether posterior parameters are symmetric."""

        for belief_update in self.belief_updates:
            with self.subTest():
                belief, _ = belief_update(
                    problem=self.linsys,
                    belief=self.prior,
                    action=self.action,
                    observation=self.observation,
                    solver_state=self.solver_state,
                )
                Ainv = belief.Ainv
                Ainv_mean = Ainv.mean.todense()
                Ainv_cov_A = Ainv.cov.A.todense()
                Ainv_cov_B = Ainv.cov.B.todense()
                self.assertAllClose(Ainv_mean, Ainv_mean.T, rtol=1e-6)
                self.assertAllClose(Ainv_cov_A, Ainv_cov_B, rtol=1e-6)
                self.assertAllClose(Ainv_cov_A, Ainv_cov_A.T, rtol=1e-6)

    def test_matrix_posterior_computation(self):
        """Test the posterior computation by the belief update against the theoretical
        expressions."""
        # TODO

    def test_matrix_inverse_posterior_computation(self):
        """Test the posterior computation by the belief update against the theoretical
        expressions."""
        # TODO
