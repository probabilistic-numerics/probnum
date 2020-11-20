"""Tests for Kronecker-type linear operators."""

import unittest

import numpy as np

from probnum import linops
from tests.testing import NumpyAssertions


class LinearOperatorKroneckerTestCase(unittest.TestCase, NumpyAssertions):
    """Test Kronecker-type operators."""

    def setUp(self):
        self.kronecker_matrices = [
            (np.array([[4, 1, 4], [2, 3, 2]]), np.array([[-1, 4], [2, 1]])),
            (np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9]]), np.array([[1, 4]])),
        ]
        self.symmkronecker_matrices = [
            (np.array([[4, 1], [2, 3]]), np.array([[-1, 4], [2, 1]])),
            (
                np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9], [1, 0, 2]]),
                np.array([[1, 4, 0], [-3, -0.4, -100], [0.18, -2, 10]]),
            ),
        ]

    def test_vec2svec_dimension(self):
        """Check faulty dimension for Q."""
        for n in [-1, 0, 1.1, np.inf, np.nan]:
            with self.subTest():
                with self.assertRaises(
                    ValueError,
                    msg="Invalid input dimension n should raise a ValueError.",
                ):
                    linops.Svec(dim=n)

    def test_symmetrize(self):
        """The Symmetrize operators should symmetrize vectors and columns of
        matrices."""
        for n in [1, 2, 3, 5, 12]:
            with self.subTest():
                x = np.random.uniform(size=n * n)
                X = np.reshape(x, (n, n))
                y = linops.Symmetrize(dim=n) @ x

                self.assertArrayEqual(
                    y.reshape(n, n), 0.5 * (X + X.T), msg="Matrix not symmetric."
                )

                Z = np.random.uniform(size=(9, 5))
                W = linops.Symmetrize(dim=3) @ Z

                self.assertArrayEqual(
                    W,
                    np.vstack([linops.Symmetrize(dim=3) @ col for col in Z.T]).T,
                    msg="Matrix columns were not symmetrized.",
                )

                self.assertArrayEqual(
                    np.shape(W),
                    np.shape(Z),
                    msg="Symmetrized matrix columns do not have the right shape.",
                )

    def test_kronecker_transpose(self):
        """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
        for A, B in self.kronecker_matrices:
            with self.subTest():
                W = linops.Kronecker(A=A, B=B)
                V = linops.Kronecker(A=A.T, B=B.T)

                self.assertAllClose(W.T.todense(), V.todense())

    def test_kronecker_explicit(self):
        """Test the Kronecker operator against explicit matrix representations."""
        for A, B in self.kronecker_matrices:
            with self.subTest():
                W = linops.Kronecker(A=A, B=B)
                AkronB = np.kron(A, B)

                self.assertAllClose(W.todense(), AkronB)

    def test_symmkronecker_todense_symmetric(self):
        """Dense matrix from symmetric Kronecker product of two symmetric matrices must
        be symmetric."""
        C = np.array([[5, 1], [1, 10]])
        D = np.array([[-2, 0.1], [0.1, 8]])
        Ws = linops.SymmetricKronecker(A=C, B=C)
        Ws_dense = Ws.todense()
        self.assertArrayEqual(
            Ws_dense,
            Ws_dense.T,
            msg="Symmetric Kronecker product of symmetric matrices is not symmetric.",
        )

    def test_symmkronecker_explicit(self):
        """Test the symmetric Kronecker operator against explicit matrix
        representations."""
        pass

    def test_symmkronecker_transpose(self):
        """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
        for A, B in self.symmkronecker_matrices:
            with self.subTest():
                W = linops.SymmetricKronecker(A=A, B=B)
                V = linops.SymmetricKronecker(A=A.T, B=B.T)

                self.assertAllClose(W.T.todense(), V.todense())

    def test_symmkronecker_commutation(self):
        """Symmetric Kronecker products fulfill A (x)_s B = B (x)_s A"""
        for A, B in self.symmkronecker_matrices:
            with self.subTest():
                W = linops.SymmetricKronecker(A=A, B=B)
                V = linops.SymmetricKronecker(A=B, B=A)

                self.assertAllClose(W.todense(), V.todense())
