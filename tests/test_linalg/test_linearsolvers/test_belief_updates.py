"""Test cases for belief updates of probabilistic linear solvers."""

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.belief_updates import (
    SymMatrixNormalLinearObsBeliefUpdate,
    WeakMeanCorrLinearObsBeliefUpdate,
)
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.linalg.linearsolvers.observation_ops import MatrixMultObservation
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class BeliefUpdateTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for belief updates of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver belief updates."""
        self.weakmeancorr_prior = WeakMeanCorrespondenceBelief.from_inverse(
            Ainv0=self.prior.Ainv.mean, problem=self.linsys
        )

        self.weakmeancorr = WeakMeanCorrLinearObsBeliefUpdate(
            problem=self.linsys,
            belief=self.weakmeancorr_prior,
            actions=self.action,
            observations=self.observation,
            solver_state=self.solver_state,
        )
        self.linear_symmetric_gaussian = SymMatrixNormalLinearObsBeliefUpdate(
            problem=self.linsys,
            belief=self.prior,
            actions=self.action,
            observations=self.observation,
            solver_state=self.solver_state,
        )
        self.belief_updates = [self.linear_symmetric_gaussian, self.weakmeancorr]

    def test_matrix_posterior_multiplication(self):
        """Test whether multiplication with the posteriors over A and Ainv returns a
        random variable with the correct shape."""
        for belief_update in self.belief_updates:
            with self.subTest():
                x, Ainv, A, b, solver_state = belief_update()
                matshape = (self.linsys.A.shape[1], 5)
                mat = self.rng.random(size=matshape)
                Amat = A @ mat
                Ainvmat = Ainv @ mat
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


class LinearSymmetricGaussianTestCase(ProbabilisticLinearSolverTestCase):
    """Test case for the linear symmetric Gaussian belief update."""

    def setUp(self) -> None:
        """Test resources for the linear Gaussian belief update."""
        self.weakmeancorr_prior = WeakMeanCorrespondenceBelief.from_matrices(
            A0=self.prior.A.mean, Ainv0=self.prior.Ainv.mean, problem=self.linsys
        )
        self.belief_updates = [
            SymMatrixNormalLinearObsBeliefUpdate(
                problem=self.linsys,
                belief=self.prior,
                actions=self.action,
                observations=self.observation,
                solver_state=self.solver_state,
            ),
            WeakMeanCorrLinearObsBeliefUpdate(
                problem=self.linsys,
                belief=self.weakmeancorr_prior,
                actions=self.action,
                observations=self.observation,
                solver_state=self.solver_state,
            ),
        ]

    @staticmethod
    def posterior_params(action, observation, prior_mean, prior_cov_factor):
        """Posterior parameters of the symmetric linear Gaussian model."""
        delta = observation - prior_mean @ action
        u = prior_cov_factor @ action / (action.T @ prior_cov_factor @ action)
        posterior_mean = (
            prior_mean + delta @ u.T + u @ delta.T - u @ action.T @ delta @ u.T
        )
        posterior_cov_factor = prior_cov_factor - prior_cov_factor @ action @ u.T
        return posterior_mean, posterior_cov_factor

    def test_symmetric_posterior_params(self):
        """Test whether posterior parameters are symmetric."""

        for belief_update in self.belief_updates:
            with self.subTest():
                x, Ainv, A, b, _ = belief_update()
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

                A1, V1 = self.posterior_params(
                    action=s, observation=y, prior_mean=A0, prior_cov_factor=V0
                )
                Ainv1, W1 = self.posterior_params(
                    action=y, observation=s, prior_mean=Ainv0, prior_cov_factor=W0
                )

                # Computation via belief update
                prior = LinearSystemBelief(
                    x=prior_Ainv @ b,
                    A=prior_A,
                    Ainv=prior_Ainv,
                    b=rvs.Constant(linsys.b),
                )
                prior.update(
                    problem=linsys,
                    observation_op=MatrixMultObservation(),
                    action=s,
                    observation=y,
                )

                self.assertAllClose(
                    A1,
                    prior.A.mean.todense(),
                    msg="The posterior mean for A does not match its definition.",
                )
                self.assertAllClose(
                    V1,
                    prior.A.cov.A.todense(),
                    msg="The posterior covariance factor for A does not match its "
                    "definition.",
                )

                self.assertAllClose(
                    Ainv1,
                    prior.Ainv.mean.todense(),
                    msg="The posterior mean for Ainv does not match its definition.",
                )
                self.assertAllClose(
                    W1,
                    prior.Ainv.cov.A.todense(),
                    msg="The posterior covariance factor for Ainv does not match its "
                    "definition.",
                )


class WeakMeanCorrLinearObsBeliefUpdateTestCase(ProbabilisticLinearSolverTestCase):
    """Test case for the linear symmetric Gaussian belief update."""

    def setUp(self) -> None:
        """Test resources for the weak mean correspondence belief update."""
        self.updated_belief = WeakMeanCorrespondenceBelief.from_scalar(
            alpha=2.5, problem=self.linsys
        )
        self.actions = self.rng.normal(size=(self.linsys.A.shape[1], 5))
        self.observations = self.linsys.A @ self.actions

        for action, observation in zip(self.actions.T, self.observations.T):
            self.updated_belief.update(
                problem=self.linsys,
                observation_op=MatrixMultObservation(),
                action=action,
                observation=observation,
            )

    @staticmethod
    def posterior_params(
        action, observation, prior_mean, prior_cov_factor, unc_scale=1.0
    ):
        """Posterior parameters of the symmetric linear Gaussian model."""
        delta = observation - prior_mean @ action
        u = prior_cov_factor @ action / (action.T @ prior_cov_factor @ action).item()
        posterior_mean = (
            prior_mean + delta @ u.T + u @ delta.T - u @ action.T @ delta @ u.T
        )
        prior_cov_factor = prior_mean @ action @ action.T @ prior_mean / (
            action.T @ prior_mean @ action
        ).item() + unc_scale * (
            np.eye(prior_mean.shape[0]) - action @ action.T / (action.T @ action).item()
        )
        posterior_cov_factor = prior_cov_factor - prior_cov_factor @ action @ u.T
        return posterior_mean, posterior_cov_factor

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
                phi = abs(self.rng.normal())
                psi = 1 / phi
                s = self.rng.normal(size=(n, 1))
                y = linsys.A @ s

                A1, V1 = self.posterior_params(
                    action=s,
                    observation=y,
                    prior_mean=A0,
                    prior_cov_factor=linsys.A,
                    unc_scale=phi,
                )
                Ainv1, W1 = self.posterior_params(
                    action=y,
                    observation=s,
                    prior_mean=Ainv0,
                    prior_cov_factor=Ainv0,
                    unc_scale=psi,
                )

                # Computation via belief update
                prior = WeakMeanCorrespondenceBelief(
                    A0=A0,
                    Ainv0=Ainv0,
                    b=linsys.b,
                    phi=phi,
                    psi=psi,
                )
                prior.update(
                    problem=linsys,
                    observation_op=MatrixMultObservation(),
                    action=s,
                    observation=y,
                )

                self.assertAllClose(
                    A1,
                    prior.A.mean.todense(),
                    msg="The posterior mean for A does not match its definition.",
                )
                self.assertAllClose(
                    V1,
                    prior.A.cov.A.todense(),
                    msg="The posterior covariance factor for A does not match its "
                    "definition.",
                )

                self.assertAllClose(
                    Ainv1,
                    prior.Ainv.mean.todense(),
                    msg="The posterior mean for Ainv does not match its definition.",
                )
                self.assertAllClose(
                    W1,
                    prior.Ainv.cov.A.todense(),
                    msg="The posterior covariance factor for Ainv does not match its "
                    "definition.",
                )

    def test_means_correspond_weakly(self):
        r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
        :math:`y`."""
        self.assertAllClose(
            np.linalg.solve(self.updated_belief.A.mean, self.observations),
            self.updated_belief.Ainv.mean @ self.observations,
        )

    def test_uncertainty_action_space_is_zero(self):
        """Test whether the uncertainty about the system matrix in the action span of
        the already explored directions is zero."""
        self.assertAllClose(
            np.zeros(self.actions), self.updated_belief.A.cov @ self.actions
        )

    def test_uncertainty_observation_space_is_zero(self):
        """Test whether the uncertainty about the inverse in the observation span of the
        already made observations is zero."""
        self.assertAllClose(
            np.zeros(self.observations),
            self.updated_belief.Ainv.cov @ self.observations,
        )
