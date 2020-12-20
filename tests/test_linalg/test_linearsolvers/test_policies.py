"""Test cases for policies of probabilistic linear solvers."""

import unittest

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    ConjugateDirectionsPolicy,
    ExploreExploitPolicy,
    LinearSolverState,
    Policy,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSolverPolicyTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for policies of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver policies."""

        # Linear system
        self.rng = np.random.default_rng()
        self.dim = 10
        _solution = self.rng.normal(size=self.dim)
        _A = random_spd_matrix(self.dim, random_state=self.rng)
        self.linsys = LinearSystem(A=_A, b=_A @ _solution, solution=_solution)

        # Prior and solver state
        Ainv0 = rvs.Normal(
            linops.ScalarMult(scalar=2.0, shape=(self.dim, self.dim)),
            linops.SymmetricKronecker(linops.Identity(self.dim)),
        )
        A0 = rvs.Normal(
            linops.ScalarMult(scalar=0.5, shape=(self.dim, self.dim)),
            linops.SymmetricKronecker(linops.Identity(self.dim)),
        )
        # Induced distribution on x via Ainv
        # Exp(x) = Ainv b, Cov(x) = 1/2 (W b'Wb + Wbb'W)
        Wb = Ainv0.cov.A @ self.linsys.b
        bWb = np.squeeze(Wb.T @ self.linsys.b)

        def _mv(x):
            return 0.5 * (bWb * Ainv0.cov.A @ x + Wb @ (Wb.T @ x))

        cov_op = linops.LinearOperator(
            shape=self.linsys.A.shape, dtype=float, matvec=_mv, matmat=_mv
        )
        x = rvs.Normal(mean=Ainv0.mean @ self.linsys.b, cov=cov_op)
        b = rvs.Constant(self.linsys.b)
        self.prior = (x, A0, Ainv0, b)
        self.solver_state = LinearSolverState(
            belief=self.prior,
            actions=[],
            observations=[],
            iteration=0,
            residual=self.linsys.A @ self.prior[0].mean - self.linsys.b,
            rayleigh_quotients=[],
            has_converged=False,
            stopping_criterion=None,
        )

        # Policies
        def custom_policy(problem, solver_state, random_state):
            return rvs.Normal(
                np.zeros(self.dim), np.identity(self.dim), random_state=random_state
            ).sample()

        self.custom_policy = Policy(
            policy=custom_policy, is_deterministic=False, random_state=self.rng
        )
        self.conj_dir_policy = ConjugateDirectionsPolicy()
        self.explore_exploit_policy = ExploreExploitPolicy(random_state=self.rng)

        self.policies = [
            self.custom_policy,
            self.conj_dir_policy,
            self.explore_exploit_policy,
        ]

    def test_returns_vector(self):
        """Test whether policies return a vector."""
        for policy in self.policies:
            with self.subTest():
                action = policy(problem=self.linsys, solver_state=self.solver_state)
                self.assertIsInstance(
                    action,
                    np.ndarray,
                    msg=f"Action {action} returned by {policy.__class__.__name__} is "
                    f"not an np.ndarray.",
                )
                self.assertTrue(
                    np.squeeze(action).ndim == 1,
                    msg=f"Action returned by {policy.__class__.__name__} has shape"
                    f" {action.shape}.",
                )


class ConjugateDirectionsPolicyTestCase(LinearSolverPolicyTestCase):
    """Test case for the conjugate directions policy."""

    def test_true_inverse_solves_problem_in_one_step(self):
        """Test whether the solver converges in one step to the solution, if the model
        over the inverse has the true inverse as a posterior mean."""
        Ainv = np.linalg.inv(self.linsys.A)
        x = self.rng.random(self.dim)
        solver_state = LinearSolverState(
            belief=(
                rvs.Constant(x),
                self.linsys.A,
                rvs.Constant(Ainv),
                rvs.Constant(self.linsys.b),
            )
        )
        action = self.conj_dir_policy(problem=self.linsys, solver_state=solver_state)

        self.assertAllClose(
            self.linsys.solution,
            x + action.ravel(),
        )

    def test_directions_are_conjugate(self):
        """Test whether the actions given by the policy are A-conjugate."""
        # TODO: use ProbabilisticLinearSolver's solve_iter function to test this

    def test_is_deterministic(self):
        """Test whether the policy is deterministic."""
        self.assertTrue(self.conj_dir_policy.is_deterministic)
        action1 = self.conj_dir_policy(
            problem=self.linsys, solver_state=self.solver_state
        )
        action2 = self.conj_dir_policy(
            problem=self.linsys, solver_state=self.solver_state
        )
        self.assertTrue(
            np.all(action1 == action2),
            msg=f"Policy returned two different actions for the same " f"input.",
        )


class ExploreExploitPolicyTestCase(LinearSolverPolicyTestCase):
    """Test case for the explore-exploit policy."""

    def test_is_stochastic(self):
        """Test whether the policy behaves stochastically."""
        self.assertFalse(self.explore_exploit_policy.is_deterministic)
        action1 = self.explore_exploit_policy(
            problem=self.linsys, solver_state=self.solver_state
        )
        action2 = self.explore_exploit_policy(
            problem=self.linsys, solver_state=self.solver_state
        )
        self.assertFalse(
            np.all(action1 == action2),
            msg=f"Policy returned the same action for two subsequent " f"evaluations.",
        )
