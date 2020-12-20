"""Test cases for belief updates of probabilistic linear solvers."""

import numpy as np

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    BeliefUpdate,
    LinearGaussianBeliefUpdate,
    LinearSolverState,
)
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class BeliefUpdateTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for belief updates of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver belief updates."""

        # Belief updates
        def custom_belief_update(solver_state: LinearSolverState):
            raise NotImplementedError

        self.custom_belief_update = BeliefUpdate(belief_update=custom_belief_update)
        self.linear_gaussian_update = LinearGaussianBeliefUpdate()


class LinearGaussianBeliefUpdateTestCase(BeliefUpdateTestCase):
    def setUp(self) -> None:
        """Test resources for the linear Gaussian belief update."""
