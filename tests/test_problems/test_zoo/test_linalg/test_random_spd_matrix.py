"""Tests for functions generating random spd matrices."""

import numpy as np
import pytest
import scipy.sparse

from probnum.problems.zoo.linalg import random_spd_matrix


class TestRandomSPDMatrix:
    """Tests for functions generating random symmetric positive definite matrices."""

    def test_dimension(self, spd_mat: np.ndarray, n: int):
        """Test whether matrix dimension matches specified dimension."""
        assert spd_mat.shape[0] == n, "Matrix dimension does not match argument."

    def test_symmetric(self, spd_mat: np.ndarray):
        """Test whether the matrix is symmetric."""
        np.testing.assert_array_equal(spd_mat, spd_mat.T), "Matrix is not symmetric."

    def test_positive_definite(self, spd_mat: np.ndarray):
        """Test whether the matrix is positive definite."""
        assert np.all(0 <= np.linalg.eigvalsh(spd_mat)), (
            "Eigenvalues are not all " "positive."
        )

    def test_spectrum_matches_given(self, n: int, random_state: np.random.RandomState):
        """Test whether the spectrum of the test problem matches the provided
        spectrum."""
        spectrum = np.sort(random_state.uniform(0.1, 1, size=n))
        spdmat = random_spd_matrix(dim=n, spectrum=spectrum, random_state=random_state)
        eigvals = np.sort(np.linalg.eigvals(spdmat))
        np.testing.assert_allclose(
            spectrum,
            eigvals,
            err_msg="Provided spectrum doesn't match actual.",
        )

    def test_negative_eigenvalues_throws_error(
        self, random_state: np.random.RandomState
    ):
        """Test whether a non-positive spectrum throws an error."""
        with pytest.raises(ValueError):
            random_spd_matrix(dim=3, spectrum=[-1, 1, 2], random_state=random_state)


class TestRandomSparseSPDMatrix:
    """Tests for functions generating random sparse symmetric positive definite
    matrices."""

    def test_dimension(self, sparse_spd_mat: scipy.sparse.spmatrix, n: int):
        """Test whether matrix dimension matches specified dimension."""
        assert sparse_spd_mat.shape[0] == n

    def test_symmetric(self, sparse_spd_mat: scipy.sparse.spmatrix):
        """Test whether the matrix is symmetric."""
        np.testing.assert_array_equal(
            sparse_spd_mat.todense(), sparse_spd_mat.T.todense()
        ), "Matrix is not symmetric."

    def test_positive_definite(self, sparse_spd_mat: scipy.sparse.spmatrix):
        """Test whether the matrix is positive definite."""
        assert np.all(0 <= np.linalg.eigvalsh(sparse_spd_mat.todense())), (
            "Eigenvalues are not all " "positive."
        )

    def test_matrix_is_sparse(
        self, sparse_spd_mat: scipy.sparse.spmatrix, sparsemat_density: float
    ):
        """Test whether the matrix has a sufficient degree of sparsity."""
        emp_density = (
            np.sum(sparse_spd_mat != 0.0) - sparse_spd_mat.shape[0]
        ) / sparse_spd_mat.shape[0] ** 2
        assert (
            (emp_density <= sparsemat_density * 2),
            f"Matrix has {emp_density}n "
            f"non-zero entries, which doesnt match the "
            f"given degree of sparsity.",
        )
