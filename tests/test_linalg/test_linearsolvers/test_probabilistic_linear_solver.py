"""Tests for the implementation of a generic probabilistic linear solver."""

import unittest

import numpy as np
import scipy.sparse.linalg

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState, ProbabilisticLinearSolver
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.linalg.linearsolvers.observation_ops import MatrixMultObservation
from probnum.linalg.linearsolvers.policies import ConjugateDirections
from probnum.linalg.linearsolvers.stop_criteria import MaxIterations, Residual
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix
from tests.testing import NumpyAssertions


class ProbabilisticLinearSolverTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for probabilistic linear solvers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Shared test resources across test cases for probabilistic linear solvers."""
        # Linear system
        cls.rng = np.random.default_rng(42)
        cls.dim = 10
        _A = random_spd_matrix(cls.dim, random_state=cls.rng)
        cls.linsys = LinearSystem.from_matrix(A=_A, random_state=cls.rng)

        # Prior and solver state
        cls.prior = LinearSystemBelief.from_inverse(
            Ainv0=linops.ScalarMult(scalar=2.0, shape=(cls.dim, cls.dim)),
            problem=cls.linsys,
        )
        cls.solver_state = LinearSolverState(
            actions=[],
            observations=[],
            iteration=0,
            residual=cls.linsys.A @ cls.prior.x.mean - cls.linsys.b,
            action_obs_innerprods=[],
            log_rayleigh_quotients=[],
            step_sizes=[],
            has_converged=False,
            stopping_criterion=None,
        )

        # Action and observation
        cls.action = cls.rng.normal(size=(cls.linsys.A.shape[1], 1))
        cls.observation = cls.rng.normal(size=(cls.linsys.A.shape[0], 1))

        # Convergence
        cls.belief_converged = LinearSystemBelief(
            rvs.Normal(mean=cls.linsys.solution, cov=10 ** -12 * np.eye(cls.dim)),
            rvs.Constant(_A),
            rvs.Constant(np.linalg.inv(_A)),
            cls.linsys.b,
        )
        cls.solver_state_converged = LinearSolverState(
            residual=cls.linsys.A @ cls.belief_converged.x.mean - cls.linsys.b
        )

    def setUp(self) -> None:
        """Test resources for custom probabilistic linear solvers."""

        # Linear system
        dim_spd = 100
        self.spd_system = LinearSystem.from_matrix(
            A=random_spd_matrix(dim=dim_spd, random_state=self.rng),
            random_state=self.rng,
        )

        self.pls = ProbabilisticLinearSolver(
            prior=LinearSystemBelief.from_solution(
                x0=self.rng.normal(size=(dim_spd,)), problem=self.spd_system
            ),
            policy=ConjugateDirections(),
            observation_op=MatrixMultObservation(),
            stopping_criteria=[MaxIterations(), Residual()],
        )

        self.solve_iterator = self.pls.solve_iterator(
            problem=self.spd_system,
            belief=self.pls.prior,
            solver_state=self.pls._init_belief_and_solver_state(
                problem=self.spd_system
            )[1],
        )

    def test_solver_state(self):
        """Test whether the solver state is consistent with the iteration."""
        for i in range(10):
            (belief, action, observation, solver_state) = next(self.solve_iterator)

            # Iteration
            self.assertEqual(solver_state.iteration, i + 1)

            # Actions
            self.assertAllClose(solver_state.actions[-1], action)

            # Observations
            self.assertAllClose(solver_state.observations[-1], observation)

            # Action - observation inner product
            self.assertAllClose(
                solver_state.action_obs_innerprods,
                np.einsum(
                    "nk,nk->k",
                    np.hstack(solver_state.actions),
                    np.hstack(solver_state.observations),
                ),
            )

            # Residual
            self.assertAllClose(
                solver_state.residual,
                self.spd_system.A @ belief.x.mean - self.spd_system.b,
                msg="Residual in solver_state does not match actual residual.",
                rtol=10 ** -6,
                atol=10 ** -12,
            )

    # TODO use pytest.xfail to mark this
    def test_solution_equivalence(self):
        """The iteratively computed solution should match the induced solution
        estimate: x_k = E[A^-1] b"""
        for i in range(10):
            (belief, action, observation, solver_state) = next(self.solve_iterator)
            # E[x] = E[A^-1] b
            self.assertAllClose(
                belief.x.mean,
                belief.Ainv.mean @ self.spd_system.b,
                rtol=1e-5,
                msg="Solution from matrix-based probabilistic linear solver "
                "does not match the estimated inverse, i.e. x != Ainv @ b ",
            )

    def test_posterior_covariance_posdef(self):
        """Posterior covariances of the output beliefs must be positive (semi-)
        definite."""
        for i in range(10):
            (belief, action, observation, solver_state) = next(self.solve_iterator)

            # Check positive definiteness
            eps = 100 * np.finfo(float).eps
            self.assertArrayLess(
                np.zeros(belief.A.shape[0]),
                np.real_if_close(np.linalg.eigvals(belief.A.cov.A.todense())) + eps,
                msg="Covariance of A not positive semi-definite.",
            )
            self.assertArrayLess(
                np.zeros(belief.A.shape[0]),
                np.real_if_close(np.linalg.eigvals(belief.Ainv.cov.A.todense())) + eps,
                msg="Covariance of Ainv not positive semi-definite.",
            )

    def test_searchdir_conjugacy(self):
        """Search directions should remain A-conjugate up to machine precision, i.e.
        s_i^T A s_j = 0 for i != j."""
        pls = ProbabilisticLinearSolver(
            prior=LinearSystemBelief.from_solution(
                x0=self.rng.normal(size=(self.spd_system.A.shape[0],)),
                problem=self.spd_system,
            ),
            policy=ConjugateDirections(),
            observation_op=MatrixMultObservation(),
            stopping_criteria=[MaxIterations(), Residual()],
        )

        _, solver_state = pls.solve(self.spd_system)
        actions = np.hstack(solver_state.actions)

        # Compute pairwise inner products in A-space
        inner_prods = actions.T @ self.spd_system.A @ actions

        # Compare against identity matrix
        self.assertAllClose(
            np.diag(np.diag(inner_prods)),
            inner_prods,
            atol=1e-7,
            msg="Search directions from solver are not A-conjugate.",
        )

    def test_posterior_mean_CG_equivalency(self):
        """The probabilistic linear solver(s) should recover CG iterates as a posterior
        mean for specific covariances."""
        maxiter = 100
        rtol = 10 ** -6
        pls = ProbabilisticLinearSolver(
            prior=WeakMeanCorrespondenceBelief.from_scalar(
                alpha=1.0, problem=self.spd_system
            ),
            policy=ConjugateDirections(),
            observation_op=MatrixMultObservation(),
            stopping_criteria=[MaxIterations(maxiter=maxiter), Residual(rtol=rtol)],
        )

        solve_iterator = pls.solve_iterator(
            problem=self.spd_system,
            belief=pls.prior,
            solver_state=self.pls._init_belief_and_solver_state(
                problem=self.spd_system
            )[1],
        )
        # Conjugate gradient method
        cg_iterates = []

        def callback_iterates_CG(xk):
            cg_iterates.append(
                np.eye(np.shape(self.spd_system.A)[0]) @ xk
            )  # identity hack to actually save different iterations

        x0 = pls.prior.x.mean
        x_cg, _ = scipy.sparse.linalg.cg(
            A=self.spd_system.A,
            b=self.spd_system.b,
            x0=x0,
            tol=rtol,
            maxiter=maxiter,
            callback=callback_iterates_CG,
        )
        cg_iters_arr = np.array([x0.squeeze()] + cg_iterates).T

        # Probabilistic linear solver
        pls_iterates = []
        belief = None
        for ind, (belief, action, observation, solver_state) in enumerate(
            solve_iterator
        ):
            pls_iterates.append(belief.x.mean)
            belief = belief

        pls_iters_arr = np.hstack([x0] + pls_iterates)
        self.assertAllClose(belief.x.mean, x_cg, rtol=10 ** -12)
        self.assertAllClose(pls_iters_arr, cg_iters_arr, rtol=10 ** -12)

    def test_multiple_rhs(self):
        """"""
