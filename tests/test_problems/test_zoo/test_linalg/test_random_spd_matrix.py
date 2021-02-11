"""Tests for functions generating random spd matrices."""

import unittest

import numpy as np

from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix
from tests.testing import NumpyAssertions


class RandomSPDMatrixTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for random spd matrices as test problems."""

    def setUp(self) -> None:
        """Define parameters and define test problems."""
        self.rng = np.random.default_rng(42)
        self.dim_list = [1, 2, 25, 100, 250]
        self.spd_matrices = [
            random_spd_matrix(dim=n, random_state=self.rng) for n in self.dim_list
        ]
        self.density = 0.01
        self.sparse_spd_matrices = [
            random_sparse_spd_matrix(dim=n, density=self.density, random_state=self.rng)
            for n in self.dim_list
        ]
        self.matrices = self.spd_matrices + self.sparse_spd_matrices

    def test_returns_ndarray(self):
        """Test whether test problems return ndarrays."""
        for mat in self.matrices:
            with self.subTest():
                self.assertIsInstance(mat, np.ndarray)

    def test_dimension(self):
        """Test whether matrix dimension matches specified dimension."""
        for mat, dim in zip(self.matrices, self.dim_list + self.dim_list):
            with self.subTest():
                self.assertEqual(
                    mat.shape[0], dim, msg="Matrix dimension does not match argument."
                )

    def test_symmetric(self):
        """Test whether the matrix is symmetric."""
        for mat in self.matrices:
            with self.subTest():
                self.assertArrayEqual(mat, mat.T, msg="Matrix is not symmetric.")

    def test_positive_definite(self):
        """Test whether the matrix is positive definite."""
        for mat in self.matrices:
            eigvals = np.linalg.eigvals(mat)
            with self.subTest():
                self.assertTrue(
                    np.all(eigvals > 0.0), msg="Eigenvalues are not all positive."
                )

    def test_spectrum_matches_given(self):
        """Test whether the spectrum of the test problem matches the provided
        spectrum."""
        dim = 10
        spectrum = np.sort(self.rng.uniform(0.1, 1, size=dim))
        spdmat = random_spd_matrix(dim=dim, spectrum=spectrum, random_state=self.rng)
        eigvals = np.sort(np.linalg.eigvals(spdmat))
        self.assertAllClose(
            spectrum,
            eigvals,
            msg="Provided spectrum doesn't match actual.",
        )

    def test_negative_eigenvalues_throws_error(self):
        """Test whether a non-positive spectrum throws an error."""
        with self.assertRaises(ValueError):
            random_spd_matrix(dim=3, spectrum=[-1, 1, 2], random_state=self.rng)

    def test_matrix_is_sparse(self):
        """Test whether the matrix has a sufficient degree of sparsity."""
        for sparsemat in self.sparse_spd_matrices:
            with self.subTest():
                emp_density = (
                    np.sum(sparsemat != 0.0) - sparsemat.shape[0]
                ) / sparsemat.shape[0] ** 2
                self.assertLess(
                    emp_density,
                    self.density * 2,
                    msg=f"Matrix has {emp_density}n "
                    f"non-zero entries, which doesnt match the "
                    f"given degree of sparsity.",
                )
