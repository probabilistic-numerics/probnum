"""Tests for linear operators."""
import itertools
import unittest

import numpy as np

from probnum import linops
from tests.testing import NumpyAssertions


class LinearOperatorArithmeticTestCase(unittest.TestCase, NumpyAssertions):
    """Test linear operator arithmetic."""

    def setUp(self):
        """Resources for tests."""
        # Random Seed
        rng = np.random.default_rng(42)

        # Scalars and arrays
        self.scalars = [0, int(1), 0.1, -4.2, np.nan, np.inf]
        self.arrays = [rng.normal(size=[5, 4]), np.array([[3, 4], [1, 5]])]

    def test_scalar_mult(self):
        """Matrix linear operator multiplication with scalars."""
        for A, alpha in list(itertools.product(self.arrays, self.scalars)):
            with self.subTest():
                Aop = linops.Matrix(A)

                self.assertAllClose((alpha * Aop).todense(), alpha * A)

    def test_addition(self):
        """Linear operator addition."""
        for A, B in list(zip(self.arrays, self.arrays)):
            with self.subTest():
                Aop = linops.Matrix(A)
                Bop = linops.Matrix(B)

                self.assertAllClose((Aop + Bop).todense(), A + B)
