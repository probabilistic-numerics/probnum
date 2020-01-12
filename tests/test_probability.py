import pytest
import numpy as np
from probnum.probability import RandomVariable, asrandvar, Normal, Dirac
from probnum.linalg.linear_operators import LinearOperator, MatrixMult, Identity, Kronecker, SymmetricKronecker


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


# def test_rv_linop_kroneckercov():
#     """Create a random variable with linear operator mean and Kronecker product covariance."""
#
#     def mv(v):
#         return np.array([2 * v[0], 3 * v[1]])
#
#     A = LinearOperator(shape=(2, 2), matvec=mv)
#     V = Kronecker(A, A)
#     RandomVariable(mean=A, cov=V)


rv = RandomVariable(distribution=Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))


@pytest.mark.parametrize("x,rv", [(np.array([2, 3]), rv), (1, rv)])
def test_rv_addition(x, rv):
    """Addition with random variables."""
    z1 = x + rv
    z2 = rv + x
    assert z1.shape == rv.shape
    assert z2.shape == rv.shape
    assert isinstance(z1, RandomVariable)
    assert isinstance(z2, RandomVariable)

@pytest.mark.parametrize("alpha", [0, int(1), .1, np.nan, np.inf])
def test_rv_scalarmult(alpha):
    """Multiplication of random variables with scalar constants."""
    x = rv
    z = alpha * x
    assert z.shape == x.shape
    assert isinstance(z, RandomVariable)

@pytest.mark.parametrize("y", [np.array([1, 2, 3])])
def test_rv_dotproduct(y):
    """Dot product of random variables with constant vectors."""
    x = rv
    z1 = np.dot(x, y)
    z2 = np.dot(y, x)
    assert z1.shape == ()
    assert z2.shape == ()
    assert isinstance(z1, RandomVariable)
    assert isinstance(z2, RandomVariable)


@pytest.mark.parametrize("A", [np.array([[1, 2], [3, 2]])])
def test_rv_matmul(A):
    """Multiplication of random variables with constant matrices."""
    x = RandomVariable(distribution=Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))
    y2 = A @ x
    assert y2.shape[0] == A.shape[0]
    assert isinstance(y2, RandomVariable)


@pytest.mark.parametrize("A", [MatrixMult(A=np.array([[1, 2], [4, 5]]))])
def test_rv_linop_matmul(A):
    """Linear operator applied to a random variable."""
    x = RandomVariable(distribution=Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))
    y = A @ x + np.array([-1, 1])


@pytest.mark.parametrize("rv1, rv2", [(RandomVariable(), RandomVariable())])
def test_different_rv_seeds(rv1, rv2):
    """Arithmetic operation between two random variables with different seeds."""
    pass
