"""Test utility functions for kernels."""

import unittest

import numpy as np

import probnum as pn
from tests.testing import NumpyAssertions


class KernelUtilsTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for utility functions for kernels."""

    def setUp(self) -> None:
        """Create different datasets and functions."""

        # Data
        rng = np.random.default_rng()

        self.data_1d_0 = rng.normal(0, 1, size=(5, 1))
        self.data_1d_1 = rng.normal(0, 1, size=(10, 1))
        self.data_2d_0 = rng.normal(0, 1, size=(5, 2))
        self.data_2d_1 = rng.normal(0, 1, size=(10, 2))

        self.datasets = zip(
            [self.data_1d_0, self.data_2d_0], [self.data_1d_1, self.data_2d_1]
        )

        # Bivariate kernel functions
        f_lin = lambda x0, x1: np.inner(x0 - 1.0, x1 - 1.0)
        f_poly = lambda x0, x1: (np.inner(x0, x1) + 1.0) ** 4
        f_expquad = lambda x0, x1: 2.0 * np.exp(-np.inner(x0 - x1) / 2)

        self.kernfuns = [f_lin, f_poly, f_expquad]

    def test_convert_into_kernel(self):
        """Test whether bivariate functions are converted into kernels."""
        with self.subTest():
            for fun in self.kernfuns:
                self.assertIsInstance(pn.askernel(fun), pn.kernels.Kernel)

    def test_vectorization_of_kernel_functions(self):
        """Test whether bivariate functions are appropriately vectorized."""
        with self.subTest():
            for fun in self.kernfuns:
                kern = pn.askernel(fun)
                for X0, X1 in self.datasets:
                    self.assertEqual(
                        kern(X0, X1).shape,
                        (np.atleast_2d(X0.shape[0]), np.atleast_2d(X1.shape[0])),
                    )
