"""Test cases for random processes."""

import unittest

import numpy as np
import scipy.spatial

import probnum.kernels as kernels
from tests.testing import NumpyAssertions


class KernelTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for kernels."""

    def setUp(self) -> None:
        """Create different datasets and kernels for the tests."""

        # Data
        rng = np.random.default_rng()

        self.data_1d_0 = rng.normal(0, 1, size=5)
        self.data_1d_1 = rng.normal(0, 1, size=10)
        self.data_2d_0 = rng.normal(0, 1, size=(5, 2))
        self.data_2d_1 = rng.normal(0, 1, size=(10, 2))

        self.datasets = zip(
            [self.data_1d_0, self.data_2d_0], [self.data_1d_1, self.data_2d_1]
        )

        # Kernels
        self.k_custom = kernels.Kernel(fun=lambda x0, x1: np.inner(x0, x1).squeeze())
        self.k_linear = kernels.Linear(constant=1.0)
        self.k_white_noise = kernels.WhiteNoise(sigma=-1.0)
        self.k_polynomial = kernels.Polynomial(constant=1.0, exponent=3)
        self.k_rbf = kernels.ExpQuad(lengthscale=1.0)

        self.kernels = [
            self.k_custom,
            self.k_linear,
            self.k_white_noise,
            self.k_polynomial,
            self.k_rbf,
        ]

    def test_shape(self):
        """Test the shape of a kernel evaluated at sets of inputs."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern in self.kernels:
                    kernshape = (X0.shape[0], X1.shape[0])
                    if kern.output_dim > 1:
                        kernshape += (kern.output_dim, kern.output_dim)
                    self.assertEqual(kern(X0, X1).shape, kernshape)

    def test_type(self):
        """Check whether a kernel evaluates to a numpy array."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern in self.kernels:
                    self.assertIsInstance(kern(X0, X1), np.ndarray)
                    self.assertIsInstance(kern(X0), np.ndarray)

    def test_kernel_matrix(self):
        """Test the computation of the kernel matrix against a naive computation."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern in self.kernels:
                    self.assertArrayEqual(
                        kern(X0, X1),
                        scipy.spatial.distance.cdist(X0, X1, metric=kern.__fun),
                    )
