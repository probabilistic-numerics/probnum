"""Tests for problems solved by probabilistic numerical methods."""

import unittest

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem, NoisyLinearSystem
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

    def test_dimension_mismatch_raises_value_error(self):
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
        """Check whether all components of the system are (reshaped to) 2D arrays or
        linear operators."""
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

    def test_shape_matches_system_components(self):
        """Test whether the linear system shape matches the system components."""
        m, n, nrhs = 8, 5, 2
        A = np.ones((m, n))
        b = np.ones((m, nrhs))
        linsys = LinearSystem(A=A, b=b)
        self.assertEqual(
            linsys.shape,
            (m, n, nrhs),
            msg=f"Linear system shape {linsys.shape} does not match "
            f"component shapes A : {A.shape}, b: {b.shape}",
        )


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


class TestNoisyLinearSystem:
    """Tests for noisy linear systems."""

    def test_sample_noisy_system_has_correct_dimensions(self):
        """Test whether sampling from a noisy linear system returns an array of tuples
        defining linear systems."""
        n = 5
        A = rvs.Normal(
            mean=linops.Identity(n),
            cov=linops.SymmetricKronecker(linops.ScalarMult(scalar=2.0, shape=(n, n))),
        )
        b = rvs.Normal(mean=np.zeros(n), cov=np.eye(n))
        linsys_rand = NoisyLinearSystem.from_randvars(A=A, b=b)
        sample_size = (3,)
        linsys_samples = linsys_rand.sample(size=sample_size)
        assert linsys_samples.shape == sample_size
        assert linsys_samples[0][0].shape == A.shape
        assert linsys_samples[0][1].shape == (b.shape[0], 1)

        linsys_single_sample = linsys_rand.sample()
        assert linsys_single_sample[0].shape == A.shape
        assert linsys_single_sample[1].shape == (b.shape[0], 1)
