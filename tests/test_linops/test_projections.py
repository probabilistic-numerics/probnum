"""Tests for projection operators."""

import unittest

import numpy as np

from probnum import linops
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions


class OrthogonalProjectionTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for orthogonal projection operators."""

    def setUp(self) -> None:
        """Test resources for orthogonal projections."""
        self.rng = np.random.default_rng(42)
        self.orthonormal_subspace_proj = linops.OrthogonalProjection(
            subspace_basis=1 / np.sqrt(5.0) * np.array([[2.0, 1.0], [-1.0, 2]]),
            is_orthonormal=True,
        )
        self.subspace_innerprod_proj = linops.OrthogonalProjection(
            subspace_basis=self.rng.normal(size=(10, 3)),
            innerprod_matrix=random_spd_matrix(dim=10, random_state=self.rng),
        )

        self.projections = [
            self.orthonormal_subspace_proj,
            self.subspace_innerprod_proj,
        ]

    def test_shape_mismatch_raises_value_error(self):
        """Test whether a mismatched shape of the basis and the matrix defining the
        inner product raises a ValueError."""
        with self.assertRaises(ValueError):
            linops.OrthogonalProjection(np.ones((5, 2)), innerprod_matrix=np.eye(6))

    def test_projecting_twice_equals_projecting_once(self):
        """Test whether applying a projection twice is identical to applying it once."""
        for proj in self.projections:
            with self.subTest():
                Q = proj.todense()
                QQ = proj @ proj.todense()
                self.assertAllClose(Q, QQ, atol=10 ** -16)

    def test_eigvals_zero_or_one(self):
        """Test whether the eigenvalues of an orthogonal projection are zero or one."""
        eigvals = self.orthonormal_subspace_proj.eigvals()
        self.assertTrue(np.all([lam in (0.0, 1.0) for lam in eigvals]))

    def test_is_orthogonal_matrix(self):
        """Test whether the matrix from the projection is orthogonal."""
        Q = self.orthonormal_subspace_proj.todense()
        Q_T = self.orthonormal_subspace_proj.T.todense()
        self.assertAllClose(np.eye(Q.shape[0]), Q @ Q_T, atol=10 ** -16)
