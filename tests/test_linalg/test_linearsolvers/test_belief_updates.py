"""Test cases for belief updates of probabilistic linear solvers."""
from typing import Optional

import numpy as np

from probnum.linalg.linearsolvers import LinearSolverState, LinearSystemBelief
from probnum.linalg.linearsolvers.belief_updates import (
    BeliefUpdate,
    LinearSymmetricGaussian,
)
from probnum.problems import LinearSystem
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class BeliefUpdateTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for belief updates of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver belief updates."""

        # Belief updates
        def custom_belief_update(
            problem: LinearSystem,
            belief: LinearSystemBelief,
            action: np.ndarray,
            observation: np.ndarray,
            solver_state: Optional[LinearSolverState] = None,
        ):
            belief.x += action

            return belief, solver_state

        self.custom_belief_update = BeliefUpdate(belief_update=custom_belief_update)
        self.linear_symmetric_gaussian = LinearSymmetricGaussian()
        self.belief_updates = [
            self.custom_belief_update,
            self.linear_symmetric_gaussian,
        ]

    def test_return_argument_types(self):
        """Test whether a belief update returns a linear system belief and solver
        state."""
        # Action and observation
        action = self.rng.normal(size=self.linsys.A.shape[1])
        observation = self.rng.normal(size=self.linsys.A.shape[0])

        for belief_update in self.belief_updates:
            with self.subTest():
                belief, solver_state = belief_update(
                    problem=self.linsys,
                    belief=self.prior,
                    action=action,
                    observation=observation,
                    solver_state=self.solver_state,
                )
                self.assertIsInstance(belief, LinearSystemBelief)
                self.assertIsInstance(solver_state, LinearSolverState)


class LinearSymmetricGaussianTestCase(BeliefUpdateTestCase):
    """Test case for the linear symmetric Gaussian belief update."""

    def setUp(self) -> None:
        """Test resources for the linear Gaussian belief update."""
        self.belief_updates = [LinearSymmetricGaussian()]
