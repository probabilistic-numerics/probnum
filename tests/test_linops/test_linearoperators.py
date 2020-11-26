"""Tests for linear operators."""
import itertools
import unittest

import numpy as np
import scipy.sparse

from probnum import linops
from tests.testing import NumpyAssertions


class LinearOperatorTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear operators."""

    def setUp(self):
        """Resources for tests."""
        # Random Seed
        np.random.seed(42)

        # Scalars, arrays and operators
        self.scalars = [0, int(1), 0.1, -4.2, np.nan, np.inf]
        self.arrays = [np.random.normal(size=[5, 4]), np.array([[3, 4], [1, 5]])]

        def mv(v):
            return np.array([2 * v[0], v[0] + 3 * v[1]])

        self.mv = mv
        self.ops = [
            linops.MatrixMult(np.array([[-1.5, 3], [0, -230]])),
            linops.LinearOperator(shape=(2, 2), matvec=mv),
            linops.Identity(shape=4),
            linops.Kronecker(
                A=linops.MatrixMult(np.array([[2, -3.5], [12, 6.5]])),
                B=linops.Identity(shape=3),
            ),
            linops.SymmetricKronecker(
                A=linops.MatrixMult(np.array([[1, -2], [-2.2, 5]])),
                B=linops.MatrixMult(np.array([[1, -3], [0, -0.5]])),
            ),
        ]

    def test_linop_construction(self):
        """Create linear operators via various construction methods."""

        # Custom linear operator
        linops.LinearOperator(shape=(2, 2), matvec=self.mv)

        # Scipy linear operator
        scipy_linop = scipy.sparse.linalg.LinearOperator(shape=(2, 2), matvec=self.mv)
        linops.aslinop(scipy_linop)


class LinearOperatorArithmeticTestCase(LinearOperatorTestCase):
    """Test linear operator arithmetic."""

    def test_scalar_mult(self):
        """Matrix linear operator multiplication with scalars."""
        for A, alpha in list(itertools.product(self.arrays, self.scalars)):
            with self.subTest():
                Aop = linops.MatrixMult(A)

                self.assertAllClose((alpha * Aop).todense(), alpha * A)

    def test_addition(self):
        """Linear operator addition."""
        for A, B in list(zip(self.arrays, self.arrays)):
            with self.subTest():
                Aop = linops.MatrixMult(A)
                Bop = linops.MatrixMult(B)

                self.assertAllClose((Aop + Bop).todense(), A + B)

    def test_matvec(self):
        """Matrix vector multiplication for linear operators."""
        np.random.seed(1)
        for op in self.ops:
            with self.subTest():
                A = op.todense()
                x = np.random.normal(size=op.shape[1])

                self.assertAllClose(A @ x, op @ x)
                self.assertAllClose(
                    A @ x[:, None],
                    op @ x[:, None],
                    msg="Matrix-vector multiplication with (n,1) vector failed.",
                )


class LinearOperatorFunctionsTestCase(LinearOperatorTestCase):
    """Test functions of linear operators."""

    def test_transpose_dense(self):
        """Test whether a transposed linear operators dense representation is equal to
        its dense representation transposed."""
        for op in self.ops:
            with self.subTest():
                A = op.todense()
                Atrans = op.T.todense()
                self.assertAllClose(Atrans, A.T)

    def test_inv_dense(self):
        """Test whether the inverse in its dense representation matches the inverse of
        the dense representation."""
        for op in self.ops:
            with self.subTest():
                try:
                    A = op.todense()
                    Ainvop = op.inv()
                    self.assertAllClose(Ainvop.todense(), np.linalg.inv(A))
                except (NotImplementedError, AttributeError):
                    pass

    def test_cond_dense(self):
        """Test whether the condition number of the linear operator matches the
        condition number of the dense representation."""
        for A in self.ops:
            with self.subTest():
                try:
                    self.assertApproxEqual(
                        A.cond(), np.linalg.cond(A.todense()), significant=7
                    )
                except NotImplementedError:
                    pass

    def test_det_dense(self):
        """Test whether the determinant of the linear operator matches the determinant
        of the dense representation."""
        for A in self.ops:
            with self.subTest():
                try:
                    self.assertApproxEqual(
                        A.det(), np.linalg.det(A.todense()), significant=7
                    )
                except NotImplementedError:
                    pass

    def test_logabsdet_dense(self):
        """Test whether the log-determinant of the linear operator matches the
        log- determinant of the dense representation."""
        for A in self.ops:
            with self.subTest():
                try:
                    self.assertApproxEqual(
                        A.logabsdet(), np.linalg.slogdet(A.todense())[1], significant=7
                    )
                except NotImplementedError:
                    pass

    def test_trace_only_square(self):
        """Test that the trace can only be computed for square matrices."""
        nonsquare_op = linops.MatrixMult(np.array([[-1.5, 3, 1], [0, -230, 0]]))
        with self.assertRaises(ValueError):
            nonsquare_op.trace()

    def test_trace_dense(self):
        """Check whether the trace matches the trace of the dense representation."""
        for A in self.ops:
            with self.subTest():
                self.assertApproxEqual(
                    A.trace(), np.trace(A.todense()).item(), significant=7
                )

    def test_adjoint(self):
        pass

    def test_todense(self):
        pass
