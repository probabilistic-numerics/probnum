"""Test cases for policies of probabilistic linear solvers."""


import numpy as np

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState, LinearSystemBelief
from probnum.linalg.linearsolvers.policies import (
    ConjugateDirectionsPolicy,
    ExploreExploitPolicy,
    Policy,
)
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class LinearSolverPolicyTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for policies of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver policies."""

        # Policies
        def custom_policy(problem, belief, random_state, solver_state=None):
            action = rvs.Normal(
                np.zeros(self.dim), np.identity(self.dim), random_state=random_state
            ).sample()
            return action, solver_state

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
        """Test whether policies return a vector of length n."""
        for policy in self.policies:
            with self.subTest():
                action, _ = policy(
                    problem=self.linsys,
                    belief=self.prior,
                    solver_state=self.solver_state,
                )
                self.assertIsInstance(
                    action,
                    np.ndarray,
                    msg=f"Action {action} returned by {policy.__class__.__name__} is "
                    "not an np.ndarray.",
                )
                self.assertTrue(
                    action.shape == (self.linsys.A.shape[1],),
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
        belief = LinearSystemBelief(
            x=rvs.Constant(x),
            A=self.linsys.A,
            Ainv=rvs.Constant(Ainv),
            b=rvs.Constant(self.linsys.b),
        )
        action, _ = self.conj_dir_policy(problem=self.linsys, belief=belief)

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
            msg="Policy returned two different actions for the same input.",
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
            msg="Policy returned the same action for two subsequent evaluations.",
        )
