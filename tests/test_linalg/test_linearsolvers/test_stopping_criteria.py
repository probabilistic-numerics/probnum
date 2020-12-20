"""Tests for stopping criteria of probabilistic linear solvers."""

import numpy as np

from probnum.linalg.linearsolvers import (
    MaxIterStoppingCriterion,
    PosteriorStoppingCriterion,
    ResidualStoppingCriterion,
    StoppingCriterion,
)
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class LinearSolverStoppingCriterionTestCase(
    ProbabilisticLinearSolverTestCase, NumpyAssertions
):
    """General test case for stopping criteria of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver stopping criteria."""

        # Stopping criteria
        def custom_stopping_criterion(problem, solver_state):
            return (
                solver_state.iteration >= 100
                or solver_state.belief[2].cov.trace() < 10 ** -3
            )

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

    def test_stop_crit_returns_bool(self):
        """Test whether stopping criteria return a boolean value."""
        for stopcrit in self.stopping_criteria:
            with self.subTest():
                has_converged = stopcrit(self.linsys, self.solver_state)
                self.assertIsInstance(has_converged, (bool, np.bool_))


class MaxIterationsTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the maximum iterations stopping criterion."""

    def test_stop_if_iter_larger_or_equal_than_maxiter(self):
        """Test if stopping criterion returns true for iteration >= maxiter."""
        for maxiter in [-1, 0, 1.0, 10, 100]:
            stopcrit = MaxIterStoppingCriterion(maxiter=maxiter)
            with self.subTest():
                has_converged = stopcrit(self.linsys, self.solver_state)
                if self.solver_state.iteration >= maxiter:
                    self.assertTrue(has_converged)
                else:
                    self.assertFalse(has_converged)


class ResidualTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the residual stopping criterion."""

    def test_stops_if_true_solution(self):
        """Test if stopping criterion returns True for exact solution."""


class PosteriorContractionTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the posterior contraction stopping criterion."""

    def test_stops_if_true_solution(self):
        """Test if stopping criterion returns True for exact solution."""
