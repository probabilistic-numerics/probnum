"""Tests for SuiteSparse matrices and related functions."""

import scipy.sparse

from probnum.problems.zoo.linalg import suitesparse_matrix


def test_downloaded_matrix_is_spmatrix(self):
    """Test whether a sparse scipy matrix is returned."""
    for mat in self.sparse_matrices:
        with self.subTest():
            self.assertIsInstance(mat, scipy.sparse.spmatrix)
