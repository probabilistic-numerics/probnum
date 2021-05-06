"""Tests for the implementation of a generic probabilistic linear solver."""

import copy
from typing import Iterator

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from probnum.linalg.solvers import ProbabilisticLinearSolver
from probnum.problems import LinearSystem, NoisyLinearSystem


class TestSolverState:
    """Test cases for the linear solver state."""

    def test_solver_state(self, linsys_spd: LinearSystem, solve_iterator: Iterator):
        """Test whether the solver state is consistent with the iteration."""
        for i in range(10):
            try:
                belief, action, observation, solver_state = next(solve_iterator)
            except StopIteration:
                break

            # Iteration
            assert (
                solver_state.info.iteration == i + 1
            ), "Solver state iteration does not match iterations performed."

            # Actions
            assert solver_state.cache.action == action

            # Observations
            assert solver_state.cache.observation == observation

            # Action - observation inner product
            np.testing.assert_allclose(
                np.array(solver_state.cache.action_observation_innerprod_list),
                np.einsum(
                    "nk,nk->k",
                    solver_state.data.actions_arr.actA,
                    solver_state.data.observations_arr.obsA,
                ),
            )

            # Belief
            assert belief == solver_state.belief
            assert solver_state.cache.belief == solver_state.belief

            # Residual
            np.testing.assert_allclose(
                solver_state.cache.residual,
                linsys_spd.A @ belief.x.mean - linsys_spd.b,
                rtol=10 ** -6,
                atol=10 ** -10,
                err_msg="Residual in solver_state does not match actual residual.",
            )


class TestProbabilisticLinearSolver:
    """Test cases for probabilistic linear solvers."""

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="This is currently not fulfilled for all PLS variants.",
    )
    def test_solution_equivalence(
        self, linsys_spd: LinearSystem, solve_iterator: Iterator
    ):
        """The iteratively computed solution should match the induced solution
        estimate: x_k = E[A^-1] b"""
        for i in range(10):
            try:
                belief, action, observation, solver_state = next(solve_iterator)
            except StopIteration:
                break

            # E[x] = E[A^-1] b
            np.testing.assert_allclose(
                belief.x.mean,
                belief.Ainv.mean @ linsys_spd.b,
                rtol=1e-5,
                err_msg="Solution from matrix-based probabilistic linear solver "
                "does not match the estimated inverse, i.e. x != Ainv @ b ",
            )

    def test_posterior_covariance_posdef(
        self, linsys_spd: LinearSystem, solve_iterator: Iterator
    ):
        """Posterior covariances of the output beliefs must be positive (semi-)
        definite."""
        for i in range(10):
            try:
                belief, action, observation, solver_state = next(solve_iterator)
            except StopIteration:
                break

            # Check positive definiteness
            eps = 10 ** 6 * np.finfo(float).eps
            assert np.all(
                scipy.linalg.eigvalsh(belief.A.cov.A.todense()) + eps >= 0
            ), "Covariance of A not positive semi-definite."
            assert np.all(
                scipy.linalg.eigvalsh(belief.Ainv.cov.A.todense()) + eps >= 0
            ), "Covariance of Ainv not positive semi-definite."


class TestConjugateDirectionsMethod:
    """Tests for probabilistic linear solvers which are conjugate direction methods."""

    def test_searchdir_conjugacy(
        self,
        conj_dir_method: ProbabilisticLinearSolver,
        linsys_spd: LinearSystem,
        n: int,
        random_state: np.random.RandomState,
    ):
        """Search directions should remain A-conjugate up to machine precision, i.e.
        s_i^T A s_j = 0 for i != j."""
        _, solver_state = conj_dir_method.solve(linsys_spd)
        actions = solver_state.data.actions_arr.actA

        # Compute pairwise inner products in A-space
        inner_prods = actions.T @ linsys_spd.A @ actions

        # Compare against identity matrix
        np.testing.assert_allclose(
            np.diag(np.diag(inner_prods)),
            inner_prods,
            atol=1e-6 * n,
            err_msg="Search directions from solver are not A-conjugate.",
        )

    def test_convergence_in_at_most_n_iterations(
        self,
        conj_dir_method: ProbabilisticLinearSolver,
        linsys_spd: LinearSystem,
        n: int,
    ):
        """Test whether the PLS takes at most n iterations, i.e. the convergence
        property of conjugate direction methods in exact arithmetic."""
        _, solver_state = conj_dir_method.solve(linsys_spd)
        assert solver_state.info.iteration <= n


class TestConjugateGradientMethod:
    """Tests for probabilistic linear solvers which are conjugate gradient methods."""

    def test_preconditioned_CG_equivalence(
        self,
        precond_conj_grad_method: ProbabilisticLinearSolver,
        linsys_spd: LinearSystem,
    ):
        """Test whether the PLS recovers preconditioned CG in posterior mean for
        specific prior beliefs."""
        solve_iterator = precond_conj_grad_method.solve_iterator(
            problem=linsys_spd,
            belief=precond_conj_grad_method.prior,
        )
        # Conjugate gradient method
        cg_iterates = []

        def callback_iterates_CG(xk):
            cg_iterates.append(copy.copy(xk[:, None]))

        x0 = precond_conj_grad_method.prior.x.mean
        x_cg, _ = scipy.sparse.linalg.cg(
            A=linsys_spd.A,
            b=linsys_spd.b,
            M=precond_conj_grad_method.prior.Ainv.mean,
            x0=x0,
            tol=precond_conj_grad_method.stopping_criteria[1].rtol,
            atol=precond_conj_grad_method.stopping_criteria[1].atol,
            maxiter=precond_conj_grad_method.stopping_criteria[0].maxiter,
            callback=callback_iterates_CG,
        )
        cg_iters_arr = np.hstack([x0] + cg_iterates)

        # Probabilistic linear solver
        pls_iterates = []
        belief = None
        for (belief, action, observation, solver_state) in solve_iterator:
            pls_iterates.append(belief.x.mean)

        pls_iters_arr = np.hstack([x0] + pls_iterates)

        # Compare iterates
        np.testing.assert_allclose(
            belief.x.mean, x_cg[:, None], atol=10 ** -4, rtol=10 ** -4
        )
        np.testing.assert_allclose(
            pls_iters_arr, cg_iters_arr, atol=10 ** -4, rtol=10 ** -4
        )

    def test_CG_equivalence(
        self, conj_grad_method: ProbabilisticLinearSolver, linsys_spd: LinearSystem
    ):
        """The probabilistic linear solver(s) should recover CG iterates as a posterior
        mean for specific prior beliefs."""
        solve_iterator = conj_grad_method.solve_iterator(
            problem=linsys_spd,
            belief=conj_grad_method.prior,
        )
        # Conjugate gradient method
        cg_iterates = []

        def callback_iterates_CG(xk):
            cg_iterates.append(copy.copy(xk[:, None]))

        x0 = conj_grad_method.prior.x.mean
        x_cg, _ = scipy.sparse.linalg.cg(
            A=linsys_spd.A,
            b=linsys_spd.b,
            x0=x0,
            tol=conj_grad_method.stopping_criteria[1].rtol,
            atol=conj_grad_method.stopping_criteria[1].atol,
            maxiter=conj_grad_method.stopping_criteria[0].maxiter,
            callback=callback_iterates_CG,
        )
        cg_iters_arr = np.hstack([x0] + cg_iterates)

        # Probabilistic linear solver
        pls_iterates = []
        belief = None
        for (belief, action, observation, solver_state) in solve_iterator:
            pls_iterates.append(belief.x.mean)

        pls_iters_arr = np.hstack([x0] + pls_iterates)

        # Compare iterates
        np.testing.assert_allclose(
            belief.x.mean, x_cg[:, None], atol=10 ** -5, rtol=10 ** -5
        )
        np.testing.assert_allclose(
            pls_iters_arr, cg_iters_arr, atol=10 ** -5, rtol=10 ** -5
        )


class TestNoisySolver:
    """Test cases for linear solvers on noisy linear systems."""

    def test_noisy_linear_system(
        self, linsys_noise: NoisyLinearSystem, noisy_solver: ProbabilisticLinearSolver
    ):
        """Test whether various different noise-corrupted linear systems are solved."""
        belief, solver_state = noisy_solver.solve(problem=linsys_noise)
        np.testing.assert_allclose(
            linsys_noise.solution, belief.x.mean, atol=10 ** -5, rtol=10 ** -5
        )
