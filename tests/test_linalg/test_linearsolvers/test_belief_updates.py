"""Test cases for belief updates of probabilistic linear solvers."""
from typing import Optional

import numpy as np

from probnum.linalg.linearsolvers.belief_updates import (
    BeliefUpdate,
    LinearSymmetricGaussian,
)
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class BeliefUpdateTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for belief updates of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver belief updates."""

        # Belief updates
        def custom_belief_update(
            belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
            action: np.ndarray,
            observation: np.ndarray,
            solver_state: Optional[
                "probnum.linalg.linearsolvers.LinearSolverState"
            ] = None,
        ):
            raise NotImplementedError

        self.custom_belief_update = BeliefUpdate(belief_update=custom_belief_update)
        self.linear_gaussian_update = LinearSymmetricGaussian()


class LinearGaussianBeliefUpdateTestCase(BeliefUpdateTestCase):
    def setUp(self) -> None:
        """Test resources for the linear Gaussian belief update."""
