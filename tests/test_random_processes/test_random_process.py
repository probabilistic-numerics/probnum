"""Test cases for random processes."""

import unittest

import numpy as np

from tests.testing import NumpyAssertions


class RandomProcessTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for random variables."""

    def setUp(self) -> None:
        """Create different random processes for tests."""

        # Seed
        np.random.seed(42)

        # Mean functions
        mean_zero = lambda x: np.zeros_like(x)

        # Covariance functions
        cov_lin = lambda x0, x1: (x0 - 1.0) @ (x1 - 1.0).T
        cov_poly = lambda x0, x1: (x0 @ x1.T) ** 3
        cov_expquad = lambda x0, x1: np.exp(
            -0.5 * np.linalg.norm(x0 - x1, ord=2, axis=0)
        )

        # Generic random processes

        # Gaussian processes


class InstantiationTestCase(RandomProcessTestCase):
    """Test random process instantiation"""


class ArithmeticTestCase(RandomProcessTestCase):
    """Test random process arithmetic"""


class ShapeTestCase(RandomProcessTestCase):
    """Test random process shape(s) and that of sample paths."""
