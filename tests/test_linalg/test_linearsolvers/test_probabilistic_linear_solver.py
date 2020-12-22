"""Tests for the implementation of a generic probabilistic linear solver."""

import os
import unittest

import numpy as np
import scipy.sparse

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState, LinearSystemBelief
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions


class ProbabilisticLinearSolverTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for probabilistic linear solvers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Shared test resources across test cases for probabilistic linear solvers."""
        # Linear system
        cls.rng = np.random.default_rng()
        cls.dim = 10
        _solution = cls.rng.normal(size=cls.dim)
        _A = random_spd_matrix(cls.dim, random_state=cls.rng)
        cls.linsys = LinearSystem(A=_A, b=_A @ _solution, solution=_solution)

        # Prior and solver state
        Ainv0 = rvs.Normal(
            linops.ScalarMult(scalar=2.0, shape=(cls.dim, cls.dim)),
            linops.SymmetricKronecker(linops.Identity(cls.dim)),
        )
        A0 = rvs.Normal(
            linops.ScalarMult(scalar=0.5, shape=(cls.dim, cls.dim)),
            linops.SymmetricKronecker(linops.Identity(cls.dim)),
        )
        # Induced distribution on x via Ainv
        # Exp(x) = Ainv b, Cov(x) = 1/2 (W b'Wb + Wbb'W)
        Wb = Ainv0.cov.A @ cls.linsys.b
        bWb = np.squeeze(Wb.T @ cls.linsys.b)

        def _mv(x):
            return 0.5 * (bWb * Ainv0.cov.A @ x + (Wb @ x) * Wb)

        def _mm(X):
            return 0.5 * (bWb * Ainv0.cov.A @ X + Wb[:, None] @ (Wb[:, None].T @ X))

        cov_op = linops.LinearOperator(
            shape=cls.linsys.A.shape, dtype=float, matvec=_mv, matmat=_mm
        )
        x = rvs.Normal(mean=Ainv0.mean @ cls.linsys.b, cov=cov_op)
        b = rvs.Constant(cls.linsys.b)
        cls.prior = LinearSystemBelief(x, A0, Ainv0, b)
        cls.solver_state = LinearSolverState(
            actions=[],
            observations=[],
            iteration=0,
            residual=cls.linsys.A @ cls.prior.x.mean - cls.linsys.b,
            log_rayleigh_quotients=[],
            step_sizes=[],
            has_converged=False,
            stopping_criterion=None,
        )

        cls.belief_converged = LinearSystemBelief(
            rvs.Normal(mean=_solution, cov=10 ** -12 * np.eye(cls.dim)),
            rvs.Constant(_A),
            rvs.Constant(np.linalg.inv(_A)),
            b,
        )
        cls.solver_state_converged = LinearSolverState(
            residual=cls.linsys.A @ cls.belief_converged.x.mean - cls.linsys.b
        )

    def setUp(self) -> None:
        """Test resources for custom probabilistic linear solvers."""

        # Linear systems
        fpath = os.path.join(os.path.dirname(__file__), "../resources")
        A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
        f = np.load(file=fpath + "/rhs_poisson.npy")
        self.poisson_linear_system = A, f

    # def test_prior_distribution_from_solution_guess(self):
    #     """When constructing prior means for A and H from a guess for the solution x0,
    #     then A_0 and H_0 should be symmetric positive definite, inverses of each other
    #     and x0=Hb should hold."""
    #     for seed in range(0, 10):
    #         with self.subTest():
    #             np.random.seed(seed)
    #
    #             # Linear system
    #             A, b = self.poisson_linear_system
    #             b = b[:, np.newaxis]
    #             x0 = np.random.randn(len(b))[:, np.newaxis]
    #
    #             if x0.T @ b < 0:
    #                 x0_true = -x0
    #             elif x0.T @ b == 0:
    #                 x0_true = np.zeros_like(b)
    #             else:
    #                 x0_true = x0
    #
    #             # Matrix-based solver
    #             smbs = probnum.linalg.MatrixBasedSolver(A=A, b=b, x0=x0)
    #             A0_mean, Ainv0_mean = smbs._construct_symmetric_matrix_prior_means(
    #                 A=A, b=b, x0=x0
    #             )
    #             A0_mean_dense = A0_mean.todense()
    #             Ainv0_mean_dense = Ainv0_mean.todense()
    #
    #             # Inverse prior mean corresponding to x0
    #             self.assertAllClose(Ainv0_mean @ b, x0_true)
    #
    #             # Inverse correspondence
    #             self.assertAllClose(
    #                 A0_mean @ Ainv0_mean @ np.eye(np.shape(A)[0]),
    #                 np.eye(np.shape(A)[0]),
    #                 atol=10 ** -8,
    #                 rtol=10 ** -8,
    #             )
    #
    #             # Symmetry
    #             self.assertAllClose(Ainv0_mean_dense, Ainv0_mean_dense.T)
    #             self.assertAllClose(A0_mean_dense, A0_mean_dense.T)
    #
    #             # Positive definiteness
    #             self.assertTrue(np.all(np.linalg.eigvals(Ainv0_mean_dense) > 0))
    #             self.assertTrue(np.all(np.linalg.eigvals(A0_mean_dense) > 0))
