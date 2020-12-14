"""Test cases for policies of probabilistic linear solvers."""

import unittest

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    ConjugateDirectionsPolicy,
    ExploreExploitPolicy,
    LinearSolverPolicy,
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

        # Belief over system
        Ainv0 = rvs.Normal(
            linops.ScalarMult(scalar=2.0, shape=(self.dim, self.dim)),
            linops.SymmetricKronecker(linops.Identity(self.dim)),
        )
        A0 = rvs.Normal(
            linops.ScalarMult(scalar=0.5, shape=(self.dim, self.dim)),
            linops.SymmetricKronecker(linops.Identity(self.dim)),
        )
        x = Ainv0 @ self.linsys.b.reshape(-1, 1)
        self.belief = (x, A0, Ainv0)

        # Policies
        def custom_policy(problem, belief, random_state):
            return rvs.Normal(
                np.zeros(self.dim), np.identity(self.dim), random_state=random_state
            ).sample()

        self.custom_policy = LinearSolverPolicy(
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
                action = policy(problem=self.linsys, belief=self.belief)
                self.assertIsInstance(
                    action,
                    np.ndarray,
                    msg=f"Action {action} returned by {policy.__class__} is not an "
                    f"np.ndarray.",
                )
                self.assertTrue(
                    np.squeeze(action).ndim == 1,
                    msg=f"Action returned by {policy.__class__} has shape"
                    f" {action.shape}.",
                )


class ConjugateDirectionsPolicyTestCase(LinearSolverPolicyTestCase):
    """Test case for the conjugate directions policy."""

    def test_true_inverse_solves_problem_in_one_step(self):
        """Test whether the solver converges in one step to the solution, if the model
        over the inverse has the true inverse as a posterior mean."""
        Ainv = np.linalg.inv(self.linsys.A)
        x = self.rng.random(self.dim)
        belief = (rvs.Constant(x), self.linsys.A, rvs.Constant(Ainv))
        action = self.conj_dir_policy(self.linsys, belief)

        self.assertAllClose(
            self.linsys.solution,
            x + action.ravel(),
        )

    def test_directions_are_conjugate(self):
        """Test whether the actions given by the policy are A-conjugate."""


class ExploreExploitPolicyTestCase(LinearSolverPolicyTestCase):
    """Test case for the explore-exploit policy."""
