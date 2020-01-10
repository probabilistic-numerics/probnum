import pytest
import numpy as np
from probnum.probability import RandomVariable, asrandvar, Normal, Dirac
from probnum.linalg.linear_operators import LinearOperator, MatrixMult, Kronecker, SymmetricKronecker


def test_rv_dtype():
    """Check the random variable types."""
    pass


@pytest.mark.parametrize("x", [0, int(1), .1, np.nan, np.inf])
def test_rv_from_number(x):
    """Create a random variable from a number."""
    asrandvar(x)


@pytest.mark.parametrize("x", [np.empty([1]), np.zeros(4), np.array([]), np.array([1, 2])])
def test_rv_from_ndarray(x):
    """Create a random variable from an array."""
    asrandvar(x)


# @pytest.mark.parametrize("A", [LinearOperator()])
# def test_rv_from_linearoperator(A):
#     """Create a random variable from a linear operator."""
#     asrandvar(A)


def test_rv_linop_kroneckercov():
    """Create a random variable with linear operator mean and Kronecker product covariance."""

    def mv(v):
        return np.array([2 * v[0], 3 * v[1]])

    A = LinearOperator(shape=(2, 2), matvec=mv)
    V = Kronecker(A, A)
    RandomVariable(mean=A, cov=V)


rv = RandomVariable(distribution=Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))


@pytest.mark.parametrize("x,y", [(np.array([2, 3]), rv),
                                 (rv, rv)])
def test_rv_addition(x, y):
    """Addition with random variables."""
    z1 = x + y
    z2 = y + x


@pytest.mark.parametrize("alpha", [0, int(1), .1, np.nan, np.inf])
def test_rv_scalarmult(alpha):
    """Multiplication of random variables with scalar constants."""
    x = RandomVariable()
    z = alpha * x


@pytest.mark.parametrize("y", [np.array([1, 2, 3])])
def test_rv_dotproduct(y):
    """Dot product of random variables with constant vectors."""
    x = RandomVariable()
    np.dot(x, y)


@pytest.mark.parametrize("A", [np.array([[1, 2], [3, 2]])])
def test_rv_matmul(A):
    """Multiplication of random variables with constant matrices."""
    x = RandomVariable(distribution=Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))
    np.matmul(asrandvar(A), x)
    asrandvar(A) @ x


@pytest.mark.parametrize("A", [MatrixMult(A=np.array([[1, 2], [4, 5]]))])
def test_rv_linop_matmul(A):
    """Linear operator applied to a random variable."""
    x = RandomVariable(distribution=Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))
    y = asrandvar(A) @ x


@pytest.mark.parametrize("rv1, rv2", [(RandomVariable(), RandomVariable())])
def test_different_rv_seeds(rv1, rv2):
    """Arithmetic operation between two random variables with different seeds."""
    pass
