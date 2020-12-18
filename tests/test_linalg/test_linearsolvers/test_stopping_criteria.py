"""Tests for stopping criteria of probabilistic linear solvers."""

import unittest

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    MaxIterStoppingCriterion,
    PosteriorStoppingCriterion,
    ResidualStoppingCriterion,
    StoppingCriterion,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSolverStoppingCriterionTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for stopping criteria of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver stopping criteria."""

        # Linear system
        self.rng = np.random.default_rng()
        self.dim = 10
        _solution = self.rng.normal(size=self.dim)
        _A = random_spd_matrix(self.dim, random_state=self.rng)
        self.linsys = LinearSystem(A=_A, b=_A @ _solution, solution=_solution)

        # Belief over system
        self.iteration = 10
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

        # Stopping criteria
        def custom_stopping_criterion(iteration, problem, belief):
            if iteration >= 100 or belief[2].cov.trace() < 10 ** -3:
                return True, "custom"
            else:
                return False, None

        self.custom_stopcrit = StoppingCriterion(
            stopping_criterion=custom_stopping_criterion
        )
        self.maxiter_stopcrit = MaxIterStoppingCriterion()
        self.residual_stopcrit = ResidualStoppingCriterion()
        self.uncertainty_stopcrit = PosteriorStoppingCriterion()

        self.stopping_criteria = [
            self.custom_stopcrit,
            self.maxiter_stopcrit,
            self.residual_stopcrit,
            self.uncertainty_stopcrit,
        ]

    def test_stop_crit_returns_bool_and_string(self):
        """Test whether stopping criteria return a bool and a string or None."""
        for stopcrit in self.stopping_criteria:

            with self.subTest():
                has_converged, criterion = stopcrit(
                    self.iteration, self.linsys, self.belief
                )
                self.assertIsInstance(has_converged, bool)
                if has_converged:
                    self.assertIsInstance(criterion, str)
                else:
                    self.assertIsNone(criterion)


class MaxIterationsTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the maximum iterations stopping criterion."""

    def test_stop_if_iter_larger_or_equal_than_maxiter(self):
        """Test if stopping criterion returns true for iteration >= maxiter."""
        for maxiter in [-1, 0, 1.0, 10, 100]:
            stopcrit = MaxIterStoppingCriterion(maxiter=maxiter)
            with self.subTest():
                has_converged, criterion = stopcrit(
                    self.iteration, self.linsys, self.belief
                )
                if self.iteration >= maxiter:
                    self.assertTrue(has_converged)
                    self.assertIsInstance(criterion, str)
                else:
                    self.assertIsNone(criterion)
                    self.assertFalse(has_converged)


class ResidualTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the residual stopping criterion."""

    def test_stops_if_true_solution(self):
        """Test if stopping criterion returns True for exact solution."""


class PosteriorContractionTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the posterior contraction stopping criterion."""

    def test_stops_if_true_solution(self):
        """Test if stopping criterion returns True for exact solution."""
