"""Test cases for belief updates of probabilistic linear solvers."""

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState
from probnum.linalg.linearsolvers.belief_updates import (
    SymMatrixNormalLinearObsBeliefUpdate,
)
from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class BeliefUpdateTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for belief updates of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver belief updates."""

        self.linear_symmetric_gaussian = SymMatrixNormalLinearObsBeliefUpdate()
        self.belief_updates = [
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
        self.belief_updates = [SymMatrixNormalLinearObsBeliefUpdate()]

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
        """Test the posterior computation of the belief update against the theoretical
        expressions."""
        # pylint : disable="too-many-locals"
        for n in [10, 50, 100]:
            with self.subTest():
                A = random_spd_matrix(dim=n, random_state=self.rng)
                b = self.rng.normal(size=(n, 1))
                linsys = LinearSystem(A, b)

                # Posterior mean and covariance factor
                A0 = random_spd_matrix(dim=n, random_state=self.rng)
                Ainv0 = random_spd_matrix(dim=n, random_state=self.rng)
                V0 = random_spd_matrix(dim=n, random_state=self.rng)
                W0 = random_spd_matrix(dim=n, random_state=self.rng)
                prior_A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(V0))
                prior_Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(W0))
                s = self.rng.normal(size=(n, 1))
                y = linsys.A @ s

                def posterior_params(action, observation, prior_mean, prior_cov_factor):
                    """Posterior parameters of the symmetric linear Gaussian model."""
                    delta = observation - prior_mean @ action
                    u = (
                        prior_cov_factor
                        @ action
                        / (action.T @ prior_cov_factor @ action)
                    )
                    posterior_mean = (
                        prior_mean
                        + delta @ u.T
                        + u @ delta.T
                        - u @ action.T @ delta @ u.T
                    )
                    posterior_cov_factor = (
                        prior_cov_factor - prior_cov_factor @ action @ u.T
                    )
                    return posterior_mean, posterior_cov_factor

                A1, V1 = posterior_params(
                    action=s, observation=y, prior_mean=A0, prior_cov_factor=V0
                )
                Ainv1, W1 = posterior_params(
                    action=y, observation=s, prior_mean=Ainv0, prior_cov_factor=W0
                )

                # Computation via belief update
                prior = LinearSystemBelief(
                    x=prior_Ainv @ b,
                    A=prior_A,
                    Ainv=prior_Ainv,
                    b=rvs.Constant(linsys.b),
                )
                belief, _ = SymMatrixNormalLinearObsBeliefUpdate()(
                    problem=linsys, belief=prior, action=s, observation=y
                )

                self.assertAllClose(
                    A1,
                    belief.A.mean.todense(),
                    msg="The posterior mean for A does not match its definition.",
                )
                self.assertAllClose(
                    V1,
                    belief.A.cov.A.todense(),
                    msg="The posterior covariance factor for A does not match its "
                    "definition.",
                )

                self.assertAllClose(
                    Ainv1,
                    belief.Ainv.mean.todense(),
                    msg="The posterior mean for Ainv does not match its definition.",
                )
                self.assertAllClose(
                    W1,
                    belief.Ainv.cov.A.todense(),
                    msg="The posterior covariance factor for Ainv does not match its "
                    "definition.",
                )
