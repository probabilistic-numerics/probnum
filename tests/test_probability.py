import itertools

import pytest
import numpy as np
import scipy.sparse

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


def test_rv_linop_kroneckercov():
    """Create a random variable with linear operator mean and Kronecker product covariance."""

    def mv(v):
        return np.array([2 * v[0], 3 * v[1]])

    A = linear_operators.LinearOperator(shape=(2, 2), matvec=mv)
    V = linear_operators.Kronecker(A, A)
    probability.RandomVariable(distribution=probability.Normal(mean=A, cov=V))


# Random variable arithmetic
arrays2d = [np.empty(2), np.zeros(2), np.array([np.inf, 1]), np.array([1, -2.5])]
matrices2d = [np.array([[1, 2], [3, 2]]), np.array([[0, 0], [1.0, -4.3]])]
linops2d = [linear_operators.MatrixMult(A=np.array([[1, 2], [4, 5]]))]
randvars2d = [
    probability.RandomVariable(distribution=probability.Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]])))]
randvars2x2 = [
    probability.Normal(mean=np.array([[-2, .3], [0, 1]]),
                       cov=linear_operators.SymmetricKronecker(A=np.eye(2), B=np.ones((2, 2))))
]


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


@pytest.mark.parametrize("rv", randvars2x2)
def test_rv_vector_product(rv):
    """Matrix-variate random variable applied to vector."""
    x = np.array([1, -4])
    y = rv @ x
    y1 = rv @ x[:, None]
    assert isinstance(y, probability.RandomVariable)
    assert isinstance(y1, probability.RandomVariable)
    np.testing.assert_equal(y.shape == (2,))
    np.testing.assert_equal(y1.shape == (2, 1))


@pytest.mark.parametrize("rv1, rv2", [(probability.RandomVariable(), probability.RandomVariable())])
def test_different_rv_seeds(rv1, rv2):
    """Arithmetic operation between two random variables with different seeds."""
    pass


# Normal distribution
np.random.seed(seed=42)
m = 7
n = 3
sparsemat = scipy.sparse.rand(m=m, n=n, density=0.1, random_state=1)
normal_params = [
    (-1, 3),
    (np.random.uniform(size=10), np.eye(10)),
    (np.array([1, -5]), linear_operators.MatrixMult(A=np.array([[2, 1], [1, -.1]]))),
    (linear_operators.MatrixMult(A=np.array([[0, -5]])), linear_operators.Identity(shape=(2, 2))),
    (np.array([[1, 2], [-3, -.4], [4, 1]]), linear_operators.Kronecker(A=np.eye(3), B=5 * np.eye(2))),
    (linear_operators.MatrixMult(A=sparsemat.todense()),
     linear_operators.Kronecker(0.1 * linear_operators.Identity(m), linear_operators.Identity(n))),
    (linear_operators.MatrixMult(A=np.random.uniform(size=(2, 2))),
     linear_operators.SymmetricKronecker(A=np.array([[1, 2], [2, 1]]), B=np.array([[5, -1], [-1, 10]])))
]


@pytest.mark.parametrize("mean,cov", [(0, [1]),
                                      (np.array([1, 2]), np.array([1, 0])),
                                      (np.array([[-1, 0], [2, 1]]), np.eye(3))])
def test_normal_dimension_mismatch(mean, cov):
    """Instantiating a normal distribution with mismatched mean and covariance should result in a ValueError."""
    with pytest.raises(ValueError):
        assert probability.Normal(mean=mean, cov=cov), "Mean and covariance mismatch in normal distribution."


@pytest.mark.parametrize("mean,cov", normal_params)
def test_normal_instantiation(mean, cov):
    """Instantiation of a normal distribution with mixed mean and cov type."""
    probability.Normal(mean=mean, cov=cov)


@pytest.mark.parametrize("mean,cov", normal_params)
def test_normal_pdf(mean, cov):
    """Evaluate pdf at random input."""
    dist = probability.Normal(mean=mean, cov=cov)
    pass


def test_normal_cdf():
    """Evaluate cdf at random input."""
    pass


@pytest.mark.parametrize("mean,cov", normal_params)
def test_sample(mean, cov):
    """Draw samples and check all sample dimensions."""
    # TODO: check dimension of each realization in dist_sample
    dist = probability.Normal(mean=mean, cov=cov, random_state=1)
    dist_sample = dist.sample(size=5)
    if not np.isscalar(dist.mean()):
        ndims_rv = len(mean.shape)
        np.testing.assert_equal(dist_sample.shape[-ndims_rv:], mean.shape,
                                err_msg="Realization shape does not match mean shape.")


@pytest.mark.parametrize("mean,cov", normal_params)
def test_sample_zero_cov(mean, cov):
    """Draw sample from distribution with zero covariance and check whether it equals the mean."""
    dist = probability.Normal(mean=mean, cov=0 * cov, random_state=1)
    dist_sample = dist.sample(size=1)
    assert_str = "Draw with covariance zero does not match mean."
    if isinstance(dist.mean(), linear_operators.LinearOperator):
        np.testing.assert_allclose(dist_sample, dist.mean().todense(), err_msg=assert_str)
    else:
        np.testing.assert_allclose(dist_sample, dist.mean(), err_msg=assert_str)


def test_symmetric_samples():
    """Samples from a normal distribution with symmetric Kronecker covariance of two symmetric matrices are
    symmetric."""
    n = 25
    A = np.random.uniform(size=(n, n))
    A = 0.5 * (A + A.T)
    dist = probability.Normal(mean=A, cov=linear_operators.SymmetricKronecker(A=np.ones((n, n)), B=np.ones((n, n))),
                              random_state=100)
    dist_sample = dist.sample(size=10)
    for B in dist_sample:
        np.testing.assert_allclose(B, B.T, atol=1e-5, rtol=1e-5,
                                   err_msg="Sample from symmetric Kronecker distribution is not symmetric.")
