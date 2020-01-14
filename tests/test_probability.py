import itertools

import pytest
import numpy as np

from probnum import probability
from probnum.linalg import linear_operators

# Random variable instantiation
scalars = [0, int(1), .1, np.nan, np.inf]
arrays = [np.empty(2), np.zeros(4), np.array([]), np.array([1, 2])]


def test_rv_dtype():
    """Check the random variable types."""
    pass


@pytest.mark.parametrize("x", scalars)
def test_rv_from_number(x):
    """Create a random variable from a number."""
    probability.asrandvar(x)


@pytest.mark.parametrize("arr", arrays)
def test_rv_from_ndarray(arr):
    """Create a random variable from an array."""
    probability.asrandvar(arr)


# @pytest.mark.parametrize("A", [LinearOperator()])
# def test_rv_from_linearoperator(A):
#     """Create a random variable from a linear operator."""
#     probability.asrandvar(A)


# def test_rv_linop_kroneckercov():
#     """Create a random variable with linear operator mean and Kronecker product covariance."""
#
#     def mv(v):
#         return np.array([2 * v[0], 3 * v[1]])
#
#     A = LinearOperator(shape=(2, 2), matvec=mv)
#     V = Kronecker(A, A)
#     probability.RandomVariable(mean=A, cov=V)


# Random variable arithmetic
arrays2d = [np.empty(2), np.zeros(2), np.array([np.inf, 1]), np.array([1, -2.5])]
matrices2d = [np.array([[1, 2], [3, 2]]), np.array([[0, 0], [1.0, -4.3]])]
linops2d = [linear_operators.MatrixMult(A=np.array([[1, 2], [4, 5]]))]
randvars2d = [
    probability.RandomVariable(distribution=probability.Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))]


@pytest.mark.parametrize("x,rv", list(itertools.product(arrays2d, randvars2d)))
def test_rv_addition(x, rv):
    """Addition with random variables."""
    z1 = x + rv
    z2 = rv + x
    assert z1.shape == rv.shape
    assert z2.shape == rv.shape
    assert isinstance(z1, probability.RandomVariable)
    assert isinstance(z2, probability.RandomVariable)


@pytest.mark.parametrize("alpha, rv", list(itertools.product(scalars, randvars2d)))
def test_rv_scalarmult(alpha, rv):
    """Multiplication of random variables with scalar constants."""
    z = alpha * rv
    assert z.shape == rv.shape
    assert isinstance(z, probability.RandomVariable)


@pytest.mark.parametrize("alpha, rv", list(itertools.product(scalars, randvars2d)))
def test_rv_broadcasting(alpha, rv):
    """Broadcasting for arrays and random variables."""
    z = alpha + rv
    z = rv - alpha
    assert z.shape == rv.shape


@pytest.mark.parametrize("x, rv", list(itertools.product([np.array([1, 2]), np.array([0, -1.4])], randvars2d)))
def test_rv_dotproduct(x, rv):
    """Dot product of random variables with constant vectors."""
    z1 = np.dot(x, rv)
    z2 = np.dot(rv, x)
    assert z1.shape == ()
    assert z2.shape == ()
    assert isinstance(z1, probability.RandomVariable)
    assert isinstance(z2, probability.RandomVariable)


@pytest.mark.parametrize("A,rv", list(itertools.product(matrices2d, randvars2d)))
def test_rv_matmul(A, rv):
    """Multiplication of random variables with constant matrices."""
    y2 = A @ rv
    assert y2.shape[0] == A.shape[0]
    assert isinstance(y2, probability.RandomVariable)


@pytest.mark.parametrize("A,rv", list(itertools.product(linops2d, randvars2d)))
def test_rv_linop_matmul(A, rv):
    """Linear operator applied to a random variable."""
    y = A @ rv + np.array([-1, 1.1])
    assert y.shape[0] == A.shape[0]


@pytest.mark.parametrize("rv1, rv2", [(probability.RandomVariable(), probability.RandomVariable())])
def test_different_rv_seeds(rv1, rv2):
    """Arithmetic operation between two random variables with different seeds."""
    pass
