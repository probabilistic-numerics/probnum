"""Tests for stopping criteria of probabilistic linear solvers."""

import numpy as np

from probnum.linalg.linearsolvers.stop_criteria import (
    MaxIterations,
    PosteriorContraction,
    Residual,
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
        def custom_stopping_criterion(problem, belief, solver_state=None):
            _has_converged = belief.Ainv.cov.trace() < 10 ** -3
            try:
                return solver_state.iteration >= 100 or _has_converged
            except AttributeError:
                return _has_converged

        self.custom_stopcrit = StoppingCriterion(
            stopping_criterion=custom_stopping_criterion
        )
        self.maxiter_stopcrit = MaxIterations()
        self.residual_stopcrit = Residual()
        self.uncertainty_stopcrit = PosteriorContraction()

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
                has_converged = stopcrit(
                    problem=self.linsys,
                    belief=self.prior,
                    solver_state=self.solver_state,
                )
                self.assertIsInstance(has_converged, (bool, np.bool_))

    def test_solver_state_none(self):
        """Test whether all stopping criteria can be computed without a solver state."""
        for stopcrit in self.stopping_criteria:
            with self.subTest():
                _ = stopcrit(
                    problem=self.linsys,
                    belief=self.prior,
                    solver_state=None,
                )


class MaxIterationsTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the maximum iterations stopping criterion."""

    def test_stop_if_iter_larger_or_equal_than_maxiter(self):
        """Test if stopping criterion returns true for iteration >= maxiter."""
        for maxiter in [-1, 0, 1.0, 10, 100]:
            stopcrit = MaxIterations(maxiter=maxiter)
            with self.subTest():
                has_converged = stopcrit(
                    problem=self.linsys,
                    belief=self.prior,
                    solver_state=self.solver_state,
                )
                if self.solver_state.iteration >= maxiter:
                    self.assertTrue(has_converged)
                else:
                    self.assertFalse(has_converged)


class ResidualTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the residual stopping criterion."""

    def test_stops_if_true_solution(self):
        """Test if stopping criterion returns True for exact solution."""
        self.assertTrue(
            self.residual_stopcrit(
                problem=self.linsys,
                belief=self.belief_converged,
                solver_state=self.solver_state_converged,
            )
        )

    def test_different_norms(self):
        """Test if stopping criterion can be computed for different norms."""
        for norm_ord in [np.inf, -np.inf, 0.5, 1, 2, 10]:
            stopcrit = Residual(norm_ord=norm_ord)
            with self.subTest():
                stopcrit(
                    problem=self.linsys,
                    belief=self.prior,
                    solver_state=self.solver_state,
                )


class PosteriorContractionTestCase(LinearSolverStoppingCriterionTestCase):
    """Test case for the posterior contraction stopping criterion."""

    def test_stops_if_true_solution(self):
        """Test if stopping criterion returns True for exact solution."""
        self.assertTrue(
            self.residual_stopcrit(
                problem=self.linsys,
                belief=self.belief_converged,
                solver_state=self.solver_state_converged,
            )
        )
