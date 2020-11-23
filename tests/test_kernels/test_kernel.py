"""Test cases for kernels."""

import unittest

import numpy as np
import scipy.spatial

import probnum.kernels as kernels
import probnum.utils as _utils
from tests.testing import NumpyAssertions


class KernelTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for kernels."""

    # pylint: disable=invalid-name

    def setUp(self) -> None:
        """Create different datasets and kernels for the tests."""

        # Data
        self.rng = np.random.default_rng(42)

        self.data_0_0 = 0.5
        self.data_1_1 = 0.75
        self.data_n_0 = self.rng.normal(0, 1, size=(5,))
        self.data_n_1 = self.rng.normal(0, 1, size=(5,))
        self.data_1xd_0 = self.rng.normal(0, 1, size=(1, 2))
        self.data_1xd_1 = self.rng.normal(0, 1, size=(1, 2))
        self.data_nxd_0 = self.rng.normal(0, 1, size=(5, 3))
        self.data_nxd_1 = self.rng.normal(0, 1, size=(10, 3))

        self.datasets = zip(
            [self.data_0_0, self.data_n_0, self.data_1xd_0, self.data_nxd_0],
            [self.data_1_1, self.data_n_1, self.data_1xd_1, self.data_nxd_1],
        )

        # Kernels
        self.kernels = [
            (kernels.Kernel, {"kernelfun": lambda x0, x1: np.inner(x0, x1).squeeze()}),
            (kernels.Linear, {"shift": 1.0}),
            (kernels.WhiteNoise, {"sigma": -1.0}),
            (kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (kernels.ExpQuad, {"lengthscale": 1.5}),
            (kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
        ]

    def test_shape(self):
        """Test the shape of a kernel evaluated at sets of inputs."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern_def in self.kernels:
                    kern = kern_def[0](
                        **kern_def[1], input_dim=np.atleast_2d(X0).shape[0]
                    )
                    kernshape = (np.atleast_2d(X0).shape[0], np.atleast_2d(X1).shape[0])
                    if kern.output_dim > 1:
                        kernshape += (kern.output_dim, kern.output_dim)
                    self.assertEqual(kern(X0, X1).shape, kernshape)

    def test_type(self):
        """Check whether a kernel evaluates to a numpy array."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern_def in self.kernels:
                    kern = kern_def[0](
                        **kern_def[1], input_dim=np.atleast_2d(X0).shape[0]
                    )
                    self.assertIsInstance(kern(X0, X1), np.ndarray)
                    self.assertIsInstance(kern(X0), np.ndarray)

    def test_kernel_matrix_against_naive(self):
        """Test the computation of the kernel matrix against a naive computation."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern_def in self.kernels:
                    kern = kern_def[0](
                        **kern_def[1], input_dim=np.atleast_2d(X0).shape[0]
                    )
                    self.assertAllClose(
                        kern(X0, X1),
                        scipy.spatial.distance.cdist(
                            np.atleast_2d(X0),
                            np.atleast_2d(X1),
                            metric=lambda x0, x1: _utils.as_numpy_scalar(
                                kern(x0, x1).squeeze()
                            ),
                        ),
                        rtol=10 ** -12,
                        atol=10 ** -12,
                    )
