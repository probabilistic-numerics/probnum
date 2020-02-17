import itertools

import pytest
import numpy as np
import scipy.sparse.linalg

from probnum.linalg import linear_operators


# Linear operator construction

def mv(v):
    return np.array([2 * v[0], v[0] + 3 * v[1]])


def test_linop_construction():
    """Create linear operators via various construction methods."""

    # Custom linear operator
    linear_operators.LinearOperator(shape=(2, 2), matvec=mv)

    # Scipy linear operator
    # scipy_linop = scipy.sparse.linalg.LinearOperator(shape=(2, 2), matvec=mv)
    # linear_operators.LinearOperator(scipy_linop)


# Linear operator arithmetic
np.random.seed(42)
scalars = [0, int(1), .1, -4.2, np.nan, np.inf]
arrays = [np.random.normal(size=[5, 4]), np.array([[3, 4],
                                                   [1, 5]])]
ops = [linear_operators.MatrixMult(np.array([[-1.5, 3],
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


@pytest.mark.parametrize("A, alpha", list(itertools.product(arrays, scalars)))
def test_scalar_mult(A, alpha):
    """Matrix linear operator multiplication with scalars."""
    Aop = linear_operators.MatrixMult(A)

    np.testing.assert_allclose((alpha * Aop).todense(), alpha * A)


@pytest.mark.parametrize("A, B", list(zip(arrays, arrays)))
def test_addition(A, B):
    """Linear operator addition"""
    Aop = linear_operators.MatrixMult(A)
    Bop = linear_operators.MatrixMult(B)

    np.testing.assert_allclose((Aop + Bop).todense(), A + B)


@pytest.mark.parametrize("op", ops)
def test_matvec(op):
    """Matrix vector multiplication for linear operators."""
    np.random.seed(1)
    A = op.todense()
    x = np.random.normal(size=op.shape[1])
    np.testing.assert_allclose(A @ x, op @ x)
    np.testing.assert_allclose(A @ x[:, None], op @ x[:, None],
                               err_msg="Matrix-vector multiplication with (n,1) vector failed.")


# Basic operations
def test_transpose():
    pass


def test_adjoint():
    pass


def test_todense():
    pass


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
#     np.testing.assert_array_equal(y[diagind, :].ravel(), np.diag(A), "Entries of diagonal are not correct.")
#
#     # Check off-diagonal entries
#     supdiagtri = np.sqrt(2) * A[np.triu_indices(n, k=1)]
#     np.testing.assert_array_equal(np.delete(y, diagind), supdiagtri,
#                                   "Off-diagonal entries are incorrect.")


@pytest.mark.parametrize("n", [-1, 0, 1.1, np.inf, np.nan])
def test_vec2svec_dimension(n):
    """Check faulty dimension for Q."""
    with pytest.raises(ValueError):
        assert linear_operators.Svec(dim=n), "Invalid input dimension n should raise a ValueError."


# @pytest.mark.parametrize("n", [1, 5, 100])
# def test_vec2svec_orthonormality(n):
#     """Check orthonormality of Q: Q^TQ = I"""
#     Q = linear_operators.Vec2Svec(dim=n)
#     np.testing.assert_allclose((Q @ Q.T).todense(),
#                                np.eye(N=int(0.5 * n * (n + 1))),
#                                err_msg="Vec2Svec does not have orthonormal rows.")
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
#     np.testing.assert_allclose(linear_operators.Vec2Svec(dim=2).todense(), Q_explicit_n2)
#     np.testing.assert_allclose(linear_operators.Vec2Svec(dim=3).todense(), Q_explicit_n3)


# Kronecker products
kronecker_matrices = [(np.array([[4, 1, 4], [2, 3, 2]]), np.array([[-1, 4], [2, 1]])),
                      (np.array([[.4, 2, .8], [-.4, 0, -.9]]), np.array([[1, 4]]))]
symmkronecker_matrices = [(np.array([[4, 1], [2, 3]]), np.array([[-1, 4], [2, 1]])),
                          (np.array([[.4, 2, .8], [-.4, 0, -.9], [1, 0, 2]]),
                           np.array([[1, 4, 0], [-3, -.4, -100], [.18, -2, 10]]))]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
def test_symmetrize(n):
    """The Symmetrize operators should symmetrize vectors and columns of matrices."""
    x = np.random.uniform(size=n*n)
    X = np.reshape(x, (n, n))
    y = linear_operators.Symmetrize(dim=n) @ x

    np.testing.assert_array_equal(y.reshape(n, n), 0.5 * (X + X.T), err_msg="Matrix not symmetric.")

    Z = np.random.uniform(size=(9, 5))
    W = linear_operators.Symmetrize(dim=3) @ Z

    np.testing.assert_array_equal(W, np.vstack([linear_operators.Symmetrize(dim=3) @ col for col in Z.T]).T,
                                  err_msg="Matrix columns were not symmetrized.")

    np.testing.assert_array_equal(np.shape(W), np.shape(Z),
                                  err_msg="Symmetrized matrix columns do not have the right shape.")


@pytest.mark.parametrize("A,B", kronecker_matrices)
def test_kronecker_transpose(A, B):
    """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
    W = linear_operators.Kronecker(A=A, B=B)
    V = linear_operators.Kronecker(A=A.T, B=B.T)

    np.testing.assert_allclose(W.T.todense(), V.todense())


@pytest.mark.parametrize("A,B", kronecker_matrices)
def test_kronecker_explicit(A, B):
    """Test the Kronecker operator against explicit matrix representations."""
    W = linear_operators.Kronecker(A=A, B=B)
    AkronB = np.kron(A, B)

    np.testing.assert_allclose(W.todense(), AkronB)


def test_symmkronecker_todense_symmetric():
    """Dense matrix from symmetric Kronecker product of two symmetric matrices must be symmetric."""
    C = np.array([[5, 1], [1, 10]])
    D = np.array([[-2, .1], [.1, 8]])
    Ws = linear_operators.SymmetricKronecker(A=C, B=C)
    Ws_dense = Ws.todense()
    np.testing.assert_array_equal(Ws_dense, Ws_dense.T,
                                  err_msg="Symmetric Kronecker product of symmetric matrices is not symmetric.")


def test_symmkronecker_explicit():
    """Test the symmetric Kronecker operator against explicit matrix representations."""
    pass


@pytest.mark.parametrize("A,B", symmkronecker_matrices)
def test_symmkronecker_transpose(A, B):
    """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
    W = linear_operators.SymmetricKronecker(A=A, B=B)
    V = linear_operators.SymmetricKronecker(A=A.T, B=B.T)

    np.testing.assert_allclose(W.T.todense(), V.todense())


@pytest.mark.parametrize("A,B", symmkronecker_matrices)
def test_symmkronecker_commutation(A, B):
    """Symmetric Kronecker products fulfill A (x)_s B = B (x)_s A"""
    W = linear_operators.SymmetricKronecker(A=A, B=B)
    V = linear_operators.SymmetricKronecker(A=B, B=A)

    np.testing.assert_allclose(W.todense(), V.todense())
