"""Tests for the random variable implementation."""

import itertools
import unittest

import numpy as np
import pytest
import scipy.stats

import probnum
from probnum import linops, randvars
from tests.testing import NumpyAssertions


def test_rng_not_generator_raises_error():

    # dummy random variable that implements a sample method.
    rv = randvars.Normal(0.0, 1.0)

    # incorrect random number generator (np.random.Generator is expected)
    rng = np.random.RandomState(seed=1)

    with pytest.raises(TypeError):
        rv.sample(rng=rng)


class RandomVariableTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for random variables."""

    def setUp(self) -> None:
        """Scalars, arrays, linear operators and random variables for tests."""
        # Seed
        self.rng = np.random.default_rng(42)

        # Random variable instantiation
        self.scalars = [0, int(1), 0.1, np.nan, np.inf]
        self.arrays = [np.empty(2), np.zeros(4), np.array([]), np.array([1, 2])]

        # Random variable arithmetic
        self.arrays2d = [
            np.empty(2),
            np.zeros(2),
            np.array([np.inf, 1]),
            np.array([1, -2.5]),
        ]
        self.matrices2d = [np.array([[1, 2], [3, 2]]), np.array([[0, 0], [1.0, -4.3]])]
        self.linops2d = [linops.Matrix(A=np.array([[1, 2], [4, 5]]))]
        self.randvars2d = [
            randvars.Normal(mean=np.array([1, 2]), cov=np.array([[2, 0], [0, 5]]))
        ]
        self.randvars2x2 = [
            randvars.Normal(
                mean=np.array([[-2, 0.3], [0, 1]]),
                cov=linops.SymmetricKronecker(A=np.eye(2), B=np.ones((2, 2))),
            ),
        ]

        self.scipyrvs = [
            scipy.stats.bernoulli(0.75),
            scipy.stats.norm(4, 2.4),
            scipy.stats.multivariate_normal(np.random.randn(10), np.eye(10)),
            scipy.stats.gamma(0.74),
            scipy.stats.dirichlet(alpha=np.array([0.1, 0.1, 0.2, 0.3])),
        ]


class InstantiationTestCase(RandomVariableTestCase):
    """Test random variable instantiation."""

    def test_rv_dtype(self):
        """Check the random variable types."""
        pass

    def test_rv_from_number(self):
        """Create a random variable from a number."""
        for x in self.scalars:
            with self.subTest():
                rv = probnum.asrandvar(x)
                self.assertIsInstance(rv, randvars.RandomVariable)

    def test_rv_from_ndarray(self):
        """Create a random variable from an array."""
        for arr in self.arrays2d:
            with self.subTest():
                rv = probnum.asrandvar(arr)
                self.assertIsInstance(rv, randvars.RandomVariable)

    # def test_rv_from_linearoperator(self):
    #     """Create a random variable from a linear operator."""
    #     for linop in linops:
    #       with self.subTest():
    #           probnum.asrandvar(A)

    def test_rv_from_scipy(self):
        """Create a random variable from a scipy random variable."""
        for scipyrv in self.scipyrvs:
            with self.subTest():
                rv = probnum.asrandvar(scipyrv)
                self.assertIsInstance(rv, randvars.RandomVariable)


class ArithmeticTestCase(RandomVariableTestCase):
    """Test random variable arithmetic and broadcasting."""

    def test_rv_addition(self):
        """Addition with random variables."""
        for (x, rv) in list(itertools.product(self.arrays2d, self.randvars2d)):
            with self.subTest():
                z1 = x + rv
                z2 = rv + x
                self.assertEqual(z1.shape, rv.shape)
                self.assertEqual(z2.shape, rv.shape)
                self.assertIsInstance(z1, randvars.RandomVariable)
                self.assertIsInstance(z2, randvars.RandomVariable)

    def test_rv_scalarmult(self):
        """Multiplication of random variables with scalar constants."""
        for (alpha, rv) in list(itertools.product(self.scalars, self.randvars2d)):
            with self.subTest():
                if np.isinf(alpha):
                    with self.assertWarns(RuntimeWarning):
                        z = alpha * rv
                else:
                    z = alpha * rv
                self.assertEqual(z.shape, rv.shape)
                self.assertIsInstance(z, randvars.RandomVariable)

    def test_rv_broadcasting(self):
        """Broadcasting for arrays and random variables."""
        for alpha, rv in list(itertools.product(self.scalars, self.randvars2d)):
            with self.subTest():
                z = alpha + rv
                self.assertEqual(z.shape, rv.shape)

                z = rv - alpha
                self.assertEqual(z.shape, rv.shape)

    def test_rv_dotproduct(self):
        """Dot product of random variables with constant vectors."""
        for x, rv in list(
            itertools.product([np.array([1, 2]), np.array([0, -1.4])], self.randvars2d)
        ):
            with self.subTest():
                z1 = rv @ x
                z2 = x @ rv
                self.assertIsInstance(z1, randvars.RandomVariable)
                self.assertIsInstance(z2, randvars.RandomVariable)
                self.assertEqual(z1.shape, ())
                self.assertEqual(z2.shape, ())

    def test_rv_matmul(self):
        """Multiplication of random variables with constant matrices."""
        for A, rv in list(itertools.product(self.matrices2d, self.randvars2d)):
            with self.subTest():
                y2 = A @ rv
                self.assertEqual(y2.shape[0], A.shape[0])
                self.assertIsInstance(y2, randvars.RandomVariable)

    def test_rv_linop_matmul(self):
        """Linear operator applied to a random variable."""
        for A, rv in list(itertools.product(self.linops2d, self.randvars2d)):
            with self.subTest():
                y = A @ rv + np.array([-1, 1.1])
                self.assertEqual(y.shape[0], A.shape[0])

    def test_rv_vector_product(self):
        """Matrix-variate random variable applied to vector."""
        for rv in self.randvars2x2:
            with self.subTest():
                x = np.array([[1], [-4]])
                y = rv @ x
                X = np.kron(np.eye(rv.shape[0]), x)
                truemean = rv.mean @ x
                truecov = X.T @ rv.cov.todense() @ X
                self.assertIsInstance(
                    y,
                    randvars.RandomVariable,
                    "The variable y does not have the correct type.",
                )
                self.assertEqual(
                    y.shape, (2, 1), "Shape of resulting random variable incorrect."
                )
                self.assertAllClose(
                    y.mean, truemean, msg="Means of random variables do not match."
                )
                self.assertAllClose(
                    y.cov.todense(),
                    truecov,
                    msg="Covariances of random variables do not match.",
                )


class ShapeTestCase(RandomVariableTestCase):
    """Test methods related to the shape of a random variable or its realizations."""

    def test_reshape(self):
        """Reshape a random variable and test for correct output shape."""
        for rv in self.randvars2x2:
            for shape in [(4, 1), (2, 2), (4,), (1, 4)]:
                with self.subTest():
                    try:
                        reshaped_rv = rv.reshape(newshape=shape)

                        self.assertEqual(reshaped_rv.shape, shape)
                        self.assertEqual(
                            reshaped_rv.sample(rng=self.rng, size=1).shape, shape
                        )
                    except NotImplementedError:
                        pass
        for rv in self.randvars2d:
            for shape in [(2, 1), (2,), (1, 2)]:
                with self.subTest():
                    try:
                        reshaped_rv = rv.reshape(newshape=shape)

                        self.assertEqual(reshaped_rv.shape, shape)
                        self.assertEqual(
                            reshaped_rv.sample(rng=self.rng, size=1).shape, shape
                        )
                    except NotImplementedError:
                        pass

    def test_sample_shape(self):
        """Sample from a random variable with different sizes and check sample
        shapes."""
        pass
