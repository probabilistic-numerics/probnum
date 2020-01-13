import pytest
from itertools import product
import numpy as np
from probnum.linalg.linear_operators import MatrixMult, Identity, Kronecker, SymmetricKronecker, _vec2svec, aslinop


# Linear operator construction
def test_linop_construction():
    """Create linear operators via various construction methods."""
    pass
    # def mv(v):
    #     return np.array([v[0], v[0] + v[1]])
    #
    # LinearOperator(shape=(2, 2), matvec=mv)


# Linear operator arithmetic
np.random.seed(42)
scalars = [0, int(1), .1, -4.2, np.nan, np.inf]
arrays = [np.random.normal(size=[5, 4]), np.array([[3, 4], [1, 5]])]
ops = [MatrixMult(np.array([[-1.5, 3], [0, -230]])),
       Identity(shape=4),
       Kronecker(A=MatrixMult(np.array([[2, -3.5], [12, 6.5]])), B=Identity(shape=2)),
       SymmetricKronecker(A=MatrixMult(np.array([[1, -2], [-2.2, 5]])), B=MatrixMult(np.array([[1, -3],
                                                                                               [0, -.5]])))]


@pytest.mark.parametrize("A, alpha", list(product(arrays, scalars)))
def test_scalar_mult(A, alpha):
    """Matrix linear operator multiplication with scalars."""
    Aop = MatrixMult(A)

    np.testing.assert_allclose((alpha * Aop).todense(), alpha * A)


@pytest.mark.parametrize("A, B", list(zip(arrays, arrays)))
def test_addition(A, B):
    """Linear operator addition"""
    Aop = MatrixMult(A)
    Bop = MatrixMult(B)

    np.testing.assert_allclose((Aop + Bop).todense(), A + B)


@pytest.mark.parametrize("op", ops)
def test_matvec(op):
    """Matrix vector multiplication for linear operators."""
    np.random.seed(1)
    A = op.todense()
    x = np.random.normal(size=op.shape[1])
    np.testing.assert_allclose(A @ x, op @ x)


# Basic operations
def test_transpose():
    pass


def test_adjoint():
    pass


def test_todense():
    pass


# Linear map Q such that svec(x) = Qvec(x).
@pytest.mark.parametrize("n", [-1, 0, 1.1, np.inf, np.nan])
def test_vec2svec_dimension(n):
    """Check faulty dimension for Q."""
    with pytest.raises(ValueError):
        assert _vec2svec(n=n), "Invalid input dimension n should raise a ValueError."


@pytest.mark.parametrize("n", [1, 5, 100])
def test_vec2svec_orthonormality(n):
    """Check orthonormality of Q: Q^TQ = I"""
    Q = _vec2svec(n=n)
    np.testing.assert_allclose((Q @ Q.T).todense(),
                               np.eye(N=int(0.5 * n * (n + 1))),
                               err_msg="Vec2Svec does not have orthonormal rows.")


def test_vec2svec_vectorization():
    """Check vectorization properties: Qvec(S) = svec(S) and vec(S) = Q^T svec(S)"""
    pass


def test_vec2svec_explicit_form():
    """Check vec2svec against some explicit matrix forms."""
    s2 = np.sqrt(2) / 2
    Q_explicit_n2 = np.array([[1, 0, 0, 0],
                              [0, s2, s2, 0],
                              [0, 0, 0, 1]])

    Q_explicit_n3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, s2, 0, s2, 0, 0, 0, 0, 0],
                              [0, 0, s2, 0, 0, 0, s2, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, s2, 0, s2, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    np.testing.assert_allclose(_vec2svec(n=2).todense(), Q_explicit_n2)
    np.testing.assert_allclose(_vec2svec(n=3).todense(), Q_explicit_n3)


# Kronecker products
kronecker_matrices = [(np.array([[4, 1, 4], [2, 3, 2]]), np.array([[-1, 4], [2, 1]])),
                      (np.array([[.4, 2, .8], [-.4, 0, -.9]]), np.array([[1, 4]]))]
symmkronecker_matrices = [(np.array([[4, 1], [2, 3]]), np.array([[-1, 4], [2, 1]])),
                          (np.array([[.4, 2, .8], [-.4, 0, -.9], [1, 0, 2]]),
                           np.array([[1, 4, 0], [-3, -.4, -100], [.18, -2, 10]]))]


@pytest.mark.parametrize("A,B", kronecker_matrices)
def test_kronecker_transpose(A, B):
    """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
    W = Kronecker(A=A, B=B)
    V = Kronecker(A=A.T, B=B.T)

    np.testing.assert_allclose(W.T.todense(), V.todense())


@pytest.mark.parametrize("A,B", kronecker_matrices)
def test_kronecker_explicit(A, B):
    """Test the Kronecker operator against explicit matrix representations."""
    W = Kronecker(A=A, B=B)
    AkronB = np.kron(A, B)

    np.testing.assert_allclose(W.todense(), AkronB)


def test_symmkronecker_todense():
    """Dense matrix from symmetric Kronecker product."""
    C = np.array([[5, 1], [2, 10]])
    Ws = SymmetricKronecker(A=C, B=C)
    Ws_dense = Ws.todense()


def test_symmkronecker_explicit():
    """Test the symmetric Kronecker operator against explicit matrix representations."""
    pass


@pytest.mark.parametrize("A,B", symmkronecker_matrices)
def test_symmkronecker_transpose(A, B):
    """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
    W = SymmetricKronecker(A=A, B=B)
    V = SymmetricKronecker(A=A.T, B=B.T)

    np.testing.assert_allclose(W.T.todense(), V.todense())


@pytest.mark.parametrize("A,B", symmkronecker_matrices)
def test_symmkronecker_commutation(A, B):
    """Symmetric Kronecker products fulfill A (x)_s B = B (x)_s A"""
    W = SymmetricKronecker(A=A, B=B)
    V = SymmetricKronecker(A=B, B=A)

    np.testing.assert_allclose(W.todense(), V.todense())
