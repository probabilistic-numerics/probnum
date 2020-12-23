"""Tests for problems solved by probabilistic numerical methods."""

import unittest

import numpy as np

import probnum.linops as linops
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSystemTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for linear systems."""

    def test_wrong_type_raises_type_error(self):
        """Test whether an unexpected type raises a TypeError."""
        A = np.eye(5)
        x = np.ones(5)
        b = np.ones(5)

        # A wrong type
        with self.assertRaises(TypeError):
            LinearSystem(A=1.0, b=b)

        # x wrong type
        with self.assertRaises(TypeError):
            LinearSystem(A=A, solution="np.ones(10)", b=b)

        # b wrong type
        with self.assertRaises(TypeError):
            LinearSystem(A=np.ones(10), solution=x, b=1.0)

        # No solution does not raise error
        LinearSystem(A=A, b=b)

    def test_wrong_dimension_mismatch_raises_value_error(self):
        """Test whether mismatched components result in a ValueError."""
        A = np.ones(shape=(8, 5))

        # A does not match b
        with self.assertRaises(ValueError):
            LinearSystem(A=A, solution=np.zeros(A.shape[1]), b=np.zeros(A.shape[0] + 1))

        # A does not match x
        with self.assertRaises(ValueError):
            LinearSystem(A=A, solution=np.zeros(A.shape[1] + 1), b=np.zeros(A.shape[0]))

        # x does not match b
        with self.assertRaises(ValueError):
            LinearSystem(
                A=A, solution=np.zeros((A.shape[1], 3)), b=np.zeros((A.shape[0], 4))
            )

    def test_system_is_two_dimensional(self):
        """Check whether all components of the system are (reshaped to) 2D arrays,
        linear operators or 2D random variables."""
        linsys = LinearSystem(A=linops.Identity(5), solution=np.ones(5), b=np.ones(5))

        self.assertEqual(linsys.A.ndim, 2)
        self.assertEqual(linsys.solution.ndim, 2)
        self.assertEqual(linsys.b.ndim, 2)

    def test_non_two_dimensional_raises_value_error(self):
        """Test whether specifying higher-dimensional arrays raise a ValueError."""
        A = np.eye(5)
        x = np.ones((5, 1))
        b = np.ones((5, 1))

        # A.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystem(A=A[:, None], b=b)

        # x.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystem(A=A, solution=x[:, None], b=b)

        # b.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystem(A=A, b=b[:, None])


class RandomLinearSystemTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the generation of random linear systems."""

    def setUp(self) -> None:
        """Test resources."""
        self.rng = np.random.default_rng()
        self.rand_spd_mat = random_spd_matrix(dim=100)
        self.random_linsys = LinearSystem.from_matrix(
            A=self.rand_spd_mat, random_state=self.rng
        )

        self.linear_systems = [self.random_linsys]

    def test_returns_linear_system(self):
        """Test whether a linear system object is returned."""
        linsys = LinearSystem.from_matrix(A=self.rand_spd_mat, random_state=self.rng)
        self.assertIsInstance(linsys, LinearSystem)

    def test_matrix_is_unchanged(self):
        """Test whether the system matrix is equal to the one given."""
        self.assertArrayEqual(self.rand_spd_mat, self.random_linsys.A)

    def test_solution_is_correct(self):
        """Test whether the solution matches the system components."""
        self.assertArrayEqual(
            self.random_linsys.A @ self.random_linsys.solution, self.random_linsys.b
        )
