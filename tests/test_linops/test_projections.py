"""Tests for projection operators."""

import unittest

import numpy as np

from probnum import linops
from tests.testing import NumpyAssertions


class OrthogonalProjectionTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for orthogonal projection operators."""

    def setUp(self) -> None:
        """Test resources for orthogonal projections."""
        self.orthonormal_subspace_proj = linops.OrthogonalProjection(
            subspace_basis=1 / np.sqrt(5.0) * np.array([[2.0, 1.0], [-1.0, 2]]),
            is_orthonormal=True,
        )

    def test_eigvals_zero_or_one(self):
        """Test whether the eigenvalues of an orthogonal projection are zero or one."""
        eigvals = self.orthonormal_subspace_proj.eigvals()
        self.assertTrue(np.all([lam in (0.0, 1.0) for lam in eigvals]))

    def test_is_orthogonal_matrix(self):
        """Test whether the matrix from the projection is orthogonal."""
        Q = self.orthonormal_subspace_proj.todense()
        Q_T = self.orthonormal_subspace_proj.T.todense()
        self.assertAllClose(np.eye(Q.shape[0]), Q @ Q_T, atol=10 ** -16)
