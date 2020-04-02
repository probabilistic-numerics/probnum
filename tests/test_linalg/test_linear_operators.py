"""Tests for linear operators."""

import itertools

import unittest
from tests.custom_assertions import NumpyAssertions
import numpy as np

from probnum.linalg import linear_operators


class LinearOperatorTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear operators."""

    def setUp(self):
        """Resources for tests."""
        # Random Seed
        np.random.seed(42)

        # Scalars, arrays and operators
        self.scalars = [0, int(1), .1, -4.2, np.nan, np.inf]
        self.arrays = [np.random.normal(size=[5, 4]), np.array([[3, 4],
                                                                [1, 5]])]

        def mv(v):
            return np.array([2 * v[0], v[0] + 3 * v[1]])

        self.mv = mv
        self.ops = [linear_operators.MatrixMult(np.array([[-1.5, 3],
                                                          [0, -230]])),
                    linear_operators.LinearOperator(shape=(2, 2), matvec=mv),
                    linear_operators.Identity(shape=4),
                    linear_operators.Kronecker(A=linear_operators.MatrixMult(np.array([[2, -3.5],
                                                                                       [12, 6.5]])),
                                               B=linear_operators.Identity(shape=2)),
                    linear_operators.SymmetricKronecker(A=linear_operators.MatrixMult(np.array([[1, -2],
                                                                                                [-2.2, 5]])),
                                                        B=linear_operators.MatrixMult(np.array([[1, -3],
                                                                                                [0, -.5]])))]

    def test_linop_construction(self):
        """Create linear operators via various construction methods."""

        # Custom linear operator
        linear_operators.LinearOperator(shape=(2, 2), matvec=self.mv)

        # Scipy linear operator
        # scipy_linop = scipy.sparse.linalg.LinearOperator(shape=(2, 2), matvec=self.mv)
        # linear_operators.LinearOperator(scipy_linop)


class LinearOperatorArithmeticTestCase(LinearOperatorTestCase):
    """Test linear operator arithmetic"""

    def test_scalar_mult(self):
        """Matrix linear operator multiplication with scalars."""
        for A, alpha in list(itertools.product(self.arrays, self.scalars)):
            with self.subTest():
                Aop = linear_operators.MatrixMult(A)

                self.assertAllClose((alpha * Aop).todense(), alpha * A)

    def test_addition(self):
        """Linear operator addition"""
        for A, B in list(zip(self.arrays, self.arrays)):
            with self.subTest():
                Aop = linear_operators.MatrixMult(A)
                Bop = linear_operators.MatrixMult(B)

                self.assertAllClose((Aop + Bop).todense(), A + B)

    def test_matvec(self):
        """Matrix vector multiplication for linear operators."""
        np.random.seed(1)
        for op in self.ops:
            with self.subTest():
                A = op.todense()
                x = np.random.normal(size=op.shape[1])

                self.assertAllClose(A @ x, op @ x)
                self.assertAllClose(A @ x[:, None], op @ x[:, None],
                                    msg="Matrix-vector multiplication with (n,1) vector failed.")

    class LinearOperatorFunctionsTestCase(LinearOperatorTestCase):
        """Test linear operator functions."""

        def test_transpose(self):
            pass

        def test_adjoint(self):
            pass

        def test_todense(self):
            pass

    class LinearOperatorKroneckerTestCase(LinearOperatorTestCase):
        """Test Kronecker-type operators."""

        def setUp(self):
            self.kronecker_matrices = [(np.array([[4, 1, 4], [2, 3, 2]]), np.array([[-1, 4], [2, 1]])),
                                       (np.array([[.4, 2, .8], [-.4, 0, -.9]]), np.array([[1, 4]]))]
            self.symmkronecker_matrices = [(np.array([[4, 1], [2, 3]]), np.array([[-1, 4], [2, 1]])),
                                           (np.array([[.4, 2, .8], [-.4, 0, -.9], [1, 0, 2]]),
                                            np.array([[1, 4, 0], [-3, -.4, -100], [.18, -2, 10]]))]

        # # Linear map Q such that svec(x) = Qvec(x).
        # @pytest.mark.parametrize("n", [1, 3, 5, 100])
        # def test_svec(n):
        #     """Test symmetric vectorization operator shape and entries."""
        #     A = np.random.normal(size=(n, n))
        #     A = 0.5 * (A + A.T)
        #     svec = linear_operators.Vec2Svec(dim=n, check_symmetric=True)
        #     y = svec @ A
        #
        #     # Check shape
        #     assert np.shape(y)[0] == int(0.5 * n * (n + 1)), "Svec(X) does not have the correct dimension."
        #
        #     # Check diagonal entries
        #     diagind = np.hstack([0, np.cumsum(np.arange(2, n + 1)[::-1])])
        #     self.assertArrayEqual(y[diagind, :].ravel(), np.diag(A), "Entries of diagonal are not correct.")
        #
        #     # Check off-diagonal entries
        #     supdiagtri = np.sqrt(2) * A[np.triu_indices(n, k=1)]
        #     self.assertArrayEqual(np.delete(y, diagind), supdiagtri,
        #                                   "Off-diagonal entries are incorrect.")

        def test_vec2svec_dimension(self):
            """Check faulty dimension for Q."""
            for n in [-1, 0, 1.1, np.inf, np.nan]:
                with self.subTest():
                    with self.assertRaises(ValueError, msg="Invalid input dimension n should raise a ValueError."):
                        linear_operators.Svec(dim=n)

        # @pytest.mark.parametrize("n", [1, 5, 100])
        # def test_vec2svec_orthonormality(n):
        #     """Check orthonormality of Q: Q^TQ = I"""
        #     Q = linear_operators.Vec2Svec(dim=n)
        #     self.assertAllClose((Q @ Q.T).todense(),
        #                                np.eye(N=int(0.5 * n * (n + 1))),
        #                                msg="Vec2Svec does not have orthonormal rows.")
        #
        #
        # def test_vec2svec_explicit_form():
        #     """Check vec2svec against some explicit matrix forms."""
        #     s2 = np.sqrt(2) / 2
        #     Q_explicit_n2 = np.array([[1, 0, 0, 0],
        #                               [0, s2, s2, 0],
        #                               [0, 0, 0, 1]])
        #
        #     Q_explicit_n3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, s2, 0, s2, 0, 0, 0, 0, 0],
        #                               [0, 0, s2, 0, 0, 0, s2, 0, 0],
        #                               [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, s2, 0, s2, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        #
        #     self.assertAllClose(linear_operators.Vec2Svec(dim=2).todense(), Q_explicit_n2)
        #     self.assertAllClose(linear_operators.Vec2Svec(dim=3).todense(), Q_explicit_n3)

        def test_symmetrize(self):
            """The Symmetrize operators should symmetrize vectors and columns of matrices."""
            for n in [1, 2, 3, 5, 12]:
                with self.subTest():
                    x = np.random.uniform(size=n * n)
                    X = np.reshape(x, (n, n))
                    y = linear_operators.Symmetrize(dim=n) @ x

                    self.assertArrayEqual(y.reshape(n, n), 0.5 * (X + X.T), msg="Matrix not symmetric.")

                    Z = np.random.uniform(size=(9, 5))
                    W = linear_operators.Symmetrize(dim=3) @ Z

                    self.assertArrayEqual(W, np.vstack([linear_operators.Symmetrize(dim=3) @ col for col in Z.T]).T,
                                          msg="Matrix columns were not symmetrized.")

                    self.assertArrayEqual(np.shape(W), np.shape(Z),
                                          msg="Symmetrized matrix columns do not have the right shape.")

        def test_kronecker_transpose(self):
            """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
            for A, B in self.kronecker_matrices:
                with self.subTest():
                    W = linear_operators.Kronecker(A=A, B=B)
                    V = linear_operators.Kronecker(A=A.T, B=B.T)

                    self.assertAllClose(W.T.todense(), V.todense())

        def test_kronecker_explicit(self):
            """Test the Kronecker operator against explicit matrix representations."""
            for A, B in self.kronecker_matrices:
                with self.subTest():
                    W = linear_operators.Kronecker(A=A, B=B)
                    AkronB = np.kron(A, B)

                    self.assertAllClose(W.todense(), AkronB)

        def test_symmkronecker_todense_symmetric(self):
            """Dense matrix from symmetric Kronecker product of two symmetric matrices must be symmetric."""
            C = np.array([[5, 1], [1, 10]])
            D = np.array([[-2, .1], [.1, 8]])
            Ws = linear_operators.SymmetricKronecker(A=C, B=C)
            Ws_dense = Ws.todense()
            self.assertArrayEqual(Ws_dense, Ws_dense.T,
                                  msg="Symmetric Kronecker product of symmetric matrices is not symmetric.")

        def test_symmkronecker_explicit(self):
            """Test the symmetric Kronecker operator against explicit matrix representations."""
            pass

        def test_symmkronecker_transpose(self):
            """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
            for A, B in self.symmkronecker_matrices:
                with self.subTest():
                    W = linear_operators.SymmetricKronecker(A=A, B=B)
                    V = linear_operators.SymmetricKronecker(A=A.T, B=B.T)

                    self.assertAllClose(W.T.todense(), V.todense())

        def test_symmkronecker_commutation(self):
            """Symmetric Kronecker products fulfill A (x)_s B = B (x)_s A"""
            for A, B in self.symmkronecker_matrices:
                with self.subTest():
                    W = linear_operators.SymmetricKronecker(A=A, B=B)
                    V = linear_operators.SymmetricKronecker(A=B, B=A)

                    self.assertAllClose(W.todense(), V.todense())


if __name__ == "__main__":
    unittest.main()
