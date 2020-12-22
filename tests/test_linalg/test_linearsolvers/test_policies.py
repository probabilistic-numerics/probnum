"""Test cases for policies of probabilistic linear solvers."""


import numpy as np

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState, LinearSystemBelief
from probnum.linalg.linearsolvers.policies import (
    ConjugateDirections,
    ExploreExploit,
    Policy,
    ThompsonSampling,
)
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class PolicyTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
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
        self.conj_dir_policy = ConjugateDirections()
        self.thompson_sampling = ThompsonSampling(random_state=self.rng)
        self.explore_exploit_policy = ExploreExploit(random_state=self.rng)

        self.policies = [
            self.custom_policy,
            self.conj_dir_policy,
            self.thompson_sampling,
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

    def test_solver_state_residual_is_none(self):
        """Test whether policies return an action without a solver state or an empty
        one."""
        empty_solver_state = LinearSolverState(
            actions=[],
            observations=[],
            iteration=10,
            residual=None,
            log_rayleigh_quotients=[],
            has_converged=False,
            stopping_criterion=None,
        )
        for policy in self.policies:
            for sstate in [None, empty_solver_state]:
                with self.subTest():
                    action, _ = policy(
                        problem=self.linsys,
                        belief=self.prior,
                        solver_state=sstate,
                    )

    def test_is_deterministic_or_stochastic(self):
        """Test whether evaluating the policy is deterministic or stochastic."""
        for policy in self.policies:
            action1, _ = policy(
                problem=self.linsys, belief=self.prior, solver_state=self.solver_state
            )
            action2, _ = policy(
                problem=self.linsys, belief=self.prior, solver_state=self.solver_state
            )

            if policy.is_deterministic:
                self.assertTrue(
                    np.all(action1 == action2),
                    msg="Policy returned two different actions for the same input.",
                )
            else:
                self.assertFalse(
                    np.all(action1 == action2),
                    msg="Policy returned the same action for two subsequent evaluations.",
                )


class ConjugateDirectionsTestCase(PolicyTestCase):
    """Test case for the conjugate directions policy."""

    def test_true_inverse_solves_problem_in_one_step(self):
        """Test whether the solver converges in one step to the solution, if the model
        over the inverse has the true inverse as a posterior mean."""
        Ainv = np.linalg.inv(self.linsys.A)
        x = self.rng.random(self.dim)
        belief = LinearSystemBelief(
            x=rvs.Constant(x),
            A=rvs.Constant(self.linsys.A),
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


class ExploreExploitTestCase(PolicyTestCase):
    """Test case for the explore-exploit policy."""


class ThompsonSamplingTestCase(PolicyTestCase):
    """Test case for the thompson sampling policy."""

    def test_belief_equals_truth_solves_problem_in_one_step(self):
        """Test whether the solver converges in one step to the solution, if the model
        for the matrix, inverse and right hand side matches the truth."""
        Ainv = np.linalg.inv(self.linsys.A)
        x = self.rng.random(self.dim)
        belief = LinearSystemBelief(
            x=rvs.Constant(x),
            A=rvs.Constant(self.linsys.A),
            Ainv=rvs.Constant(Ainv),
            b=rvs.Constant(self.linsys.b),
        )
        action, _ = self.thompson_sampling(problem=self.linsys, belief=belief)

        self.assertAllClose(
            self.linsys.solution,
            x + action.ravel(),
        )
