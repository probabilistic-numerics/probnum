"""Tests for observation operators of probabilistic linear solvers."""

import numpy as np

from probnum.linalg.linearsolvers import (
    LinearSolverState,
    MatrixMultObservation,
    ObservationOperator,
)
from probnum.problems import LinearSystem
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class ObservationOperatorTestCase(ProbabilisticLinearSolverTestCase, NumpyAssertions):
    """General test case for observation operators of probabilistic linear solvers."""

    def setUp(self) -> None:
        """Test resources for observation operators."""

        # Solver state
        self.action = self.rng.normal(size=self.dim)
        self.solver_state.actions = [self.action]

        # Observation operators
        def custom_obs(problem: LinearSystem, solver_state: LinearSolverState):
            return solver_state.actions[-1] @ (problem.A @ solver_state.actions[-1])

        self.custom_obs = ObservationOperator(observation_op=custom_obs)
        self.matmult_obs = MatrixMultObservation()
        self.observation_ops = [self.custom_obs, self.matmult_obs]

    def test_observation_is_vector_or_float(self):
        """Test whether observation operators return a vector."""
        for obs_op in self.observation_ops:
            with self.subTest():
                observation = obs_op(
                    problem=self.linsys, solver_state=self.solver_state
                )
                self.assertIsInstance(
                    observation,
                    (np.ndarray, float, np.float_),
                    msg=f"Observation {observation} returned by"
                    f" {obs_op.__class__.__name__} "
                    f"is not an np.ndarray.",
                )


class MatrixMultObservationTestCase(ObservationOperatorTestCase):
    """Test case for matrix-vector product observations."""

    def test_observation_is_matrix_vector_product(self):
        """Test whether the observation is given by a matrix-vector multiplication with
        the system matrix."""
        A_copy = self.linsys.A.copy()  # in case evaluation of A is stochastic
        self.assertArrayEqual(
            self.matmult_obs(self.linsys, self.solver_state), A_copy @ self.action
        )
