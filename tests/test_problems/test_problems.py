"""Tests for problems solved by probabilistic numerical methods."""

import unittest

import numpy as np

from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions


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
