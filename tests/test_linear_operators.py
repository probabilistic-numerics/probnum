import pytest
import numpy as np
from probnum.linalg import linear_operators as linops


def test_linop_construction():
    """Create linear operators via various construction methods."""
    pass
    # def mv(v):
    #     return np.array([v[0], v[0] + v[1]])
    #
    # linops.LinearOperator(shape=(2, 2), matvec=mv)


def test_matvec():
    """Matrix vector multiplication."""
    A = 2 * np.eye(5)
    Aop = linops.aslinop(A)
    np.allclose(A, Aop @ np.eye(5))


# Linear map Q such that svec(x) = Qvec(x).
@pytest.mark.parametrize("n", [-1, 0, 1.1])
def test_vec2svec_dimension(n):
    """Check faulty dimension for Q."""
    with pytest.raises(ValueError):
        assert linops._vec2svec(n=n), "Invalid input dimension n should raise a ValueError."


@pytest.mark.parametrize("n", [1, 5, 100])
def test_vec2svec_orthonormality(n):
    """Check orthonormality of Q: Q^TQ = I"""
    Q = linops._vec2svec(n=n)
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

    np.testing.assert_allclose(linops._vec2svec(n=2).todense(), Q_explicit_n2)
    np.testing.assert_allclose(linops._vec2svec(n=3).todense(), Q_explicit_n3)


# Kronecker products
def test_kronecker_transpose():
    """Kronecker product transpose property: (A (x) B)^T = A^T (x) B^T."""
    pass


def test_kronecker_explicit():
    """Test the Kronecker operator against explicit matrix representations."""
    pass


def test_symmkronecker_todense():
    """Dense matrix from Kronecker product."""
    C = np.array([[5, 1], [2, 10]])
    Ws = linops.SymmetricKronecker(A=C, B=C)

    # in order to plot both heatmaps on the same scale, find the plotting range
    Ws_dense = Ws.todense()

def test_symmkronecker_explicit():
    """Test the symmetric Kronecker operator against explicit matrix representations."""
    pass
