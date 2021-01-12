"""Test cases for linear system beliefs."""

import unittest

import numpy as np
import scipy.linalg

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSystemBeliefTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear system beliefs."""

    def setUp(self) -> None:
        """Test resources for linear system beliefs."""
        self.rng = np.random.default_rng(42)
        self.linsys = LinearSystem.from_matrix(
            A=random_spd_matrix(dim=10), random_state=self.rng
        )
        self.belief_classes = [LinearSystemBelief, WeakMeanCorrespondenceBelief]

    def test_dimension_mismatch_raises_value_error(self):
        """Test whether mismatched components result in a ValueError."""
        m, n, nrhs = 5, 3, 2
        A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
        Ainv = A
        x = rvs.Normal(mean=np.zeros((n, nrhs)), cov=np.eye(n * nrhs))
        b = rvs.Constant(np.ones((m, nrhs)))

        # A does not match b
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A, Ainv=Ainv, x=x, b=rvs.Constant(np.ones((m + 1, nrhs)))
            )

        # A does not match x
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=Ainv,
                x=rvs.Normal(mean=np.zeros((n + 1, nrhs)), cov=np.eye((n + 1) * nrhs)),
                b=b,
            )

        # x does not match b
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=Ainv,
                x=rvs.Normal(mean=np.zeros((n, nrhs + 1)), cov=np.eye(n * (nrhs + 1))),
                b=b,
            )

        # A does not match Ainv
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=rvs.Normal(mean=np.ones((m + 1, n)), cov=np.eye((m + 1) * n)),
                x=x,
                b=b,
            )

    def test_beliefs_are_two_dimensional(self):
        """Check whether all beliefs over quantities of interest are 2 dimensional."""
        m, n = 5, 3
        A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
        Ainv = A
        x = rvs.Normal(mean=np.zeros(n), cov=np.eye(n))
        b = rvs.Constant(np.ones(m))
        belief = LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b)

        self.assertEqual(belief.A.ndim, 2)
        self.assertEqual(belief.Ainv.ndim, 2)
        self.assertEqual(belief.x.ndim, 2)
        self.assertEqual(belief.b.ndim, 2)

    def test_non_two_dimensional_raises_value_error(self):
        """Test whether specifying higher-dimensional random variables raise a
        ValueError."""
        A = rvs.Constant(np.eye(5))
        Ainv = rvs.Constant(np.eye(5))
        x = rvs.Constant(np.ones((5, 1)))
        b = rvs.Constant(np.ones((5, 1)))

        # A.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A[:, None], Ainv=Ainv, x=x, b=b)

        # Ainv.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A, Ainv=Ainv[:, None], x=x, b=b)

        # x.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A, Ainv=Ainv, x=x[:, None], b=b)

        # b.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b[:, None])

    def test_induced_solution_has_correct_distribution(self):
        """Test whether the induced distribution over the solution from a belief over
        the inverse is correct."""
        Ainv0 = random_spd_matrix(dim=self.linsys.A.shape[0], random_state=self.rng)
        W = random_spd_matrix(dim=self.linsys.A.shape[0], random_state=self.rng)
        Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=W))

        belief = LinearSystemBelief.from_inverse(Ainv0=Ainv, problem=self.linsys)

        self.assertAllClose(
            belief.x.mean,
            belief.Ainv.mean @ self.linsys.b,
            msg="Induced belief over the solution has an inconsistent mean.",
        )
        Wb = W @ self.linsys.b
        bWb = (Wb.T @ self.linsys.b).item()
        Sigma = 0.5 * (bWb * W + Wb @ Wb.T)
        self.assertAllClose(
            belief.x.cov.todense(),
            Sigma,
            msg="Induced belief over the solution has an inconsistent covariance.",
        )

    # Classmethod tests
    def test_from_solution_array(self):
        """Test whether a linear system belief can be created from a solution estimate
        given as an array."""
        x0 = self.rng.normal(size=self.linsys.A.shape[1])
        for belief_class in self.belief_classes:
            with self.subTest():
                belief_class.from_solution(x0=x0, problem=self.linsys)

    def test_from_solution_generates_consistent_inverse_belief(self):
        """Test whether the belief for the inverse generated from a solution guess
        matches the belief for the solution."""
        x0 = self.rng.normal(size=(self.linsys.A.shape[1], 1))
        for belief_class in self.belief_classes:
            with self.subTest():
                belief = belief_class.from_solution(x0=x0, problem=self.linsys)

                self.assertAllClose(belief.x.mean, belief.Ainv.mean @ self.linsys.b)

    def test_from_solution_creates_better_initialization(self):
        """Test whether if possible, a better initial value x0' is constructed from
        x0."""
        # Linear System
        linsys = LinearSystem(
            A=np.array([[4, 2, -6, 4], [2, 2, -3, 1], [-6, -3, 13, 0], [4, 1, 0, 30]]),
            solution=np.array([2, 0, -1, 2]),
            b=np.array([22, 9, -25, 68]),
        )

        x0_list = []

        # <b, x0> < 0
        x0_list.append(-linsys.b)

        # <b, x0> = 0, b != 0
        x0_list.append(np.array([0.5, -1, 0, -1 / 34])[:, None])
        self.assertAlmostEqual((x0_list[1].T @ linsys.b).item(), 0.0)

        for x0 in x0_list:
            with self.subTest():
                belief = LinearSystemBelief.from_solution(x0=x0, problem=linsys)

                self.assertGreater(
                    (belief.x.mean.T @ linsys.b).item(),
                    0.0,
                    msg="Inner product <x0, b>="
                    f"{(belief.x.mean.T @ linsys.b).item():.4f} is not positive.",
                )
                error_x0 = (
                    (linsys.solution - x0).T @ linsys.A @ (linsys.solution - x0)
                ).item()
                error_x1 = (
                    (linsys.solution - belief.x.mean).T
                    @ linsys.A
                    @ (linsys.solution - belief.x.mean)
                ).item()
                self.assertLess(
                    error_x1,
                    error_x0,
                    msg="Initialization for the solution x0 is not better in A-norm "
                    "than the user-specified one.",
                )

        # b = 0
        linsys_homogeneous = LinearSystem(A=linsys.A, b=np.zeros_like(linsys.b))
        belief = LinearSystemBelief.from_solution(
            x0=np.ones_like(linsys.b), problem=linsys_homogeneous
        )
        self.assertAllClose(belief.x.mean, np.zeros_like(linsys.b))

    def test_from_matrix_array(self):
        """Test whether a linear system belief can be created from a system matrix
        estimate given as an array."""
        A0 = self.rng.normal(size=self.linsys.A.shape)
        LinearSystemBelief.from_matrix(A0=A0, problem=self.linsys)

    def test_from_inverse_array(self):
        """Test whether a linear system belief can be created from an inverse estimate
        given as an array."""
        Ainv0 = self.rng.normal(size=self.linsys.A.shape)
        LinearSystemBelief.from_inverse(Ainv0=Ainv0, problem=self.linsys)

    def test_from_matrices_arrays(self):
        """Test whether a linear system belief can be created from an estimate of the
        system matrix and its inverse given as an arrays."""
        A0 = self.rng.normal(size=self.linsys.A.shape)
        Ainv0 = self.rng.normal(size=self.linsys.A.shape)
        for belief_class in self.belief_classes:
            with self.subTest():
                belief_class.from_matrices(A0=A0, Ainv0=Ainv0, problem=self.linsys)


class WeakMeanCorrespondenceBeliefTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the weak mean correspondence belief."""

    def setUp(self) -> None:
        """Test resources for the weak mean correspondence belief."""
        self.rng = np.random.default_rng(42)
        self.linsys = LinearSystem.from_matrix(
            A=random_spd_matrix(dim=10), random_state=self.rng
        )
        self.actions = self.rng.normal(size=(self.linsys.A.shape[1], 5))
        self.observations = self.linsys.A @ self.actions
        self.A0 = linops.ScalarMult(scalar=2.0, shape=self.linsys.A.shape)
        self.Ainv0 = self.A0.inv()
        self.phi = 2.0
        self.psi = 0.25
        self.belief = WeakMeanCorrespondenceBelief(
            A0=self.A0,
            Ainv0=self.Ainv0,
            b=self.linsys.b,
            phi=self.phi,
            psi=self.psi,
            actions=self.actions,
            observations=self.observations,
        )

    def test_means_correspond_weakly(self):
        r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
        :math:`y`."""
        self.assertAllClose(
            np.linalg.solve(self.belief.A.mean.todense(), self.observations),
            self.belief.Ainv.mean @ self.observations,
        )

    def test_system_matrix_uncertainty_in_action_span(self):
        """Test whether the covariance factor W_0^A of the model for A acts like the
        true A in the span of the actions, i.e. if W_0^A S = Y."""
        self.assertAllClose(self.observations, self.belief.A.cov.A @ self.actions)

    def test_inverse_uncertainty_in_observation_span(self):
        """Test whether the covariance factor W_0^H of the model for Ainv acts like its
        prior mean in the span of the observations, i.e. if W_0^H Y = H_0 Y."""
        self.assertAllClose(
            self.Ainv0 @ self.observations, self.belief.Ainv.cov.A @ self.observations
        )

    def test_uncertainty_action_null_space_is_phi(self):
        r"""Test whether the uncertainty in the null space <S>^\perp is
        given by the uncertainty scale parameter phi for a scalar system matrix A."""
        scalar_linsys = LinearSystem.from_matrix(
            A=5.0 * np.eye(10), random_state=self.rng
        )
        belief = WeakMeanCorrespondenceBelief(
            A0=self.A0,
            Ainv0=self.Ainv0,
            b=self.linsys.b,
            phi=self.phi,
            psi=self.psi,
            actions=self.actions,
            observations=scalar_linsys.A @ self.actions,
        )

        action_null_space = scipy.linalg.null_space(self.actions.T)

        self.assertAllClose(
            action_null_space.T @ (belief.A.cov.A @ action_null_space),
            self.phi
            * np.eye(
                self.actions.shape[1],
            ),
            atol=10 ** -15,
            rtol=10 ** -15,
        )

    def test_uncertainty_observation_null_space_is_psi(self):
        r"""Test whether the uncertainty in the null space <Y>^\perp is
        given by the uncertainty scale parameter psi for a scalar prior mean."""
        A0 = linops.ScalarMult(scalar=2.0, shape=self.linsys.A.shape)
        belief = WeakMeanCorrespondenceBelief(
            A0=A0,
            Ainv0=A0.inv(),
            b=self.linsys.b,
            phi=self.phi,
            psi=self.psi,
            actions=self.actions,
            observations=self.observations,
        )

        observation_null_space = scipy.linalg.null_space(self.observations.T)

        self.assertAllClose(
            observation_null_space.T @ (belief.Ainv.cov.A @ observation_null_space),
            self.psi * np.eye(self.observations.shape[1]),
            atol=10 ** -16,
        )

    def test_conjugate_actions_covariance(self):
        """Test whether the covariance for conjugate actions matches a naively computed
        one."""
        # Compute conjugate actions via Cholesky decomposition: S' = L^{-T}S
        actions = np.diag(self.rng.normal(size=self.linsys.A.shape[0]))[:, 0:5]
        chol = scipy.linalg.cholesky(self.linsys.A)
        conj_actions = scipy.linalg.solve_triangular(chol, actions)
        observations = self.linsys.A @ conj_actions

        # Naive covariance factors
        W0_A = observations @ np.linalg.solve(
            conj_actions.T @ observations, observations.T
        ) + self.phi * (
            np.eye(self.linsys.A.shape[0])
            - linops.OrthogonalProjection(subspace_basis=conj_actions).todense()
        )
        W0_Ainv = (
            self.psi * np.eye(self.linsys.A.shape[0])
            + (self.Ainv0.scalar - self.psi)
            * linops.OrthogonalProjection(subspace_basis=observations).todense()
        )

        belief = WeakMeanCorrespondenceBelief(
            A0=self.A0,
            Ainv0=self.Ainv0,
            b=self.linsys.b,
            phi=self.phi,
            psi=self.psi,
            actions=conj_actions,
            observations=observations,
            action_obs_innerprods=np.einsum("nk,nk->k", conj_actions, observations),
        )

        self.assertAllClose(
            belief.A.cov.A.todense(),
            W0_A,
            msg="Covariance factor of the A model does not match "
            "naively computed one.",
        )
        self.assertAllClose(
            belief.Ainv.cov.A.todense(),
            W0_Ainv,
            msg="Covariance factor of the Ainv model does not match "
            "naively computed one.",
        )

    def test_dense_inverse_prior_mean(self):
        """Teset whether the covariance for the inverse model with a dense prior mean
        matches a naively computed one."""
        Ainv0 = random_spd_matrix(dim=self.linsys.A.shape[0], random_state=self.rng)

        belief = WeakMeanCorrespondenceBelief(
            A0=self.A0,
            Ainv0=Ainv0,
            b=self.linsys.b,
            phi=self.phi,
            psi=self.psi,
            actions=self.actions,
            observations=self.observations,
        )

        W0_Ainv = Ainv0 @ linops.OrthogonalProjection(
            subspace_basis=self.observations, innerprod_matrix=Ainv0
        ).todense() + self.psi * (
            np.eye(self.linsys.A.shape[0])
            - linops.OrthogonalProjection(subspace_basis=self.observations).todense()
        )

        self.assertAllClose(
            belief.Ainv.cov.A.todense(),
            W0_Ainv,
            msg="Covariance factor of the Ainv model does not match "
            "naively computed one.",
        )

    def test_no_data_prior(self):
        """Test whether for no actions or observations the prior means and covariance
        are correct."""
        belief = WeakMeanCorrespondenceBelief(
            A0=self.A0,
            Ainv0=self.Ainv0,
            b=self.linsys.b,
            phi=self.phi,
            psi=self.psi,
        )
        # Means
        self.assertEqual(belief.A.mean, self.A0)
        self.assertEqual(belief.Ainv.mean, self.Ainv0)

        # Covariances
        self.assertIsInstance(belief.A.cov.A, linops.ScalarMult)
        self.assertEqual(belief.A.cov.A.scalar, self.phi)

        self.assertIsInstance(belief.Ainv.cov.A, linops.ScalarMult)
        self.assertEqual(belief.Ainv.cov.A.scalar, self.psi)

    # Classmethod tests
    def test_from_matrix_satisfies_mean_correspondence(self):
        """Test whether for a belief constructed from an approximate system matrix, the
        prior mean of the inverse model corresponds."""
        A0 = linops.ScalarMult(scalar=5.0, shape=self.linsys.A.shape)
        belief = WeakMeanCorrespondenceBelief.from_matrix(A0=A0, problem=self.linsys)
        self.assertAllClose(belief.Ainv.mean.inv().todense(), belief.A.mean.todense())

    def test_from_inverse_satisfies_mean_correspondence(self):
        """Test whether for a belief constructed from an approximate inverse, the prior
        mean of the system matrix model corresponds."""
        Ainv0 = linops.ScalarMult(scalar=5.0, shape=self.linsys.A.shape)
        belief = WeakMeanCorrespondenceBelief.from_inverse(
            Ainv0=Ainv0, problem=self.linsys
        )
        self.assertAllClose(belief.Ainv.mean.inv().todense(), belief.A.mean.todense())

    def test_belief_construction_inefficient_inversion_raises_error(self):
        """Test whether when a belief is constructed and the prior means are given as
        np.ndarrays, an error is raised."""
        M = self.rng.normal(size=self.linsys.A.shape)
        with self.assertRaises(TypeError):
            WeakMeanCorrespondenceBelief.from_matrix(A0=M, problem=self.linsys)

        with self.assertRaises(TypeError):
            WeakMeanCorrespondenceBelief.from_inverse(Ainv0=M, problem=self.linsys)

    def test_from_scalar(self):
        """Test whether a linear system belief can be created from a scalar."""
        WeakMeanCorrespondenceBelief.from_scalar(alpha=2.5, problem=self.linsys)

    def test_from_scalar_nonpositive_raises_value_error(self):
        """Test whether attempting to construct a weak mean correspondence belief from a
        non-positive scalar results in a ValueError."""
        for alpha in [-1.0, -10, 0.0, 0]:
            with self.assertRaises(ValueError):
                WeakMeanCorrespondenceBelief.from_scalar(
                    alpha=alpha, problem=self.linsys
                )

    # Hyperparameters
    def test_uncertainty_calibration(self):
        """"""
        pass  # TODO

    def test_uncertainty_calibration_error(self):
        """Test if the available uncertainty calibration procedures affect the error of
        the returned solution."""
        # tol = 10 ** -6
        # A, b, x_true = self.rbf_kernel_linear_system
        #
        # for calib_method in [None, 0, "adhoc", "weightedmean", "gpkern"]:
        #     with self.subTest():
        #         x_est, Ahat, Ainvhat, info = linalg.problinsolve(
        #             A=A, b=b, calibration=calib_method
        #         )
        #         self.assertLessEqual(
        #             (x_true - x_est.mean).T @ A @ (x_true - x_est.mean),
        #             tol,
        #             msg="Estimated solution not sufficiently close to true solution.",
        #         )
