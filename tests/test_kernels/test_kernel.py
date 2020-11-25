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
        self.data_n_0 = self.rng.normal(0, 1, size=(1,))
        self.data_n_1 = self.rng.normal(0, 1, size=(1,))
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
                    # Define and evaluate kernel
                    kern = kern_def[0](
                        **kern_def[1], input_dim=np.atleast_2d(X0).shape[1]
                    )
                    kernmat = kern(X0, X1)

                    # Check shape
                    X0 = np.asarray(X0)
                    X1 = np.asarray(X1)
                    if (X0.ndim == 0 and X1.ndim == 0) or (
                        X0.ndim == 1 and X1.ndim == 1
                    ):
                        kern_shape = ()
                    else:
                        kern_shape = (X0.shape[0], X1.shape[0])
                    if kern.output_dim > 1:
                        kern_shape += (kern.output_dim, kern.output_dim)

                    self.assertTupleEqual(
                        kernmat.shape,
                        kern_shape,
                        msg=f"Kernel {type(kern)} does not have the right shape if "
                        f"evaluated at inputs of x0.shape={X0.shape} and x1.shape="
                        f"{X1.shape}.",
                    )

    def test_type(self):
        """Check whether a kernel evaluates to a numpy scalar or array."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern_def in self.kernels:
                    kern = kern_def[0](
                        **kern_def[1], input_dim=np.atleast_2d(X0).shape[1]
                    )
                    for kernmat in (kern(X0), kern(X0, X1)):
                        self.assertTrue(
                            isinstance(kernmat, np.ndarray)
                            or np.isscalar(
                                kernmat,
                            ),
                            msg=f"{type(kernmat)} is neither a scalar nor an "
                            f"numpy.ndarray.",
                        )

    def test_kernel_matrix_against_naive(self):
        """Test the computation of the kernel matrix against a naive computation."""
        with self.subTest():
            for (X0, X1) in self.datasets:
                for kern_def in self.kernels:
                    kern = kern_def[0](
                        **kern_def[1], input_dim=np.atleast_2d(X0).shape[1]
                    )
                    self.assertAllClose(
                        kern(X0, X1),
                        scipy.spatial.distance.cdist(
                            np.atleast_2d(X0),
                            np.atleast_2d(X1),
                            metric=lambda x0, x1, k=kern: _utils.as_numpy_scalar(
                                k(x0, x1).squeeze()
                            ),
                        ),
                        rtol=10 ** -12,
                        atol=10 ** -12,
                    )

    def test_misshaped_input(self):
        """Test whether misshaped/mismatched input throws an error."""
        kern = kernels.Kernel(
            input_dim=2, output_dim=1, kernelfun=lambda x0, x1: np.array(1.0)
        )
        datasets = [
            (1, 1),
            (1.0, np.array([1.0, 0.0])),
            (np.array([1.0, 0.0, 0.2]), np.array([1.0, 0.0, 2.3])),
            (np.array([[1.0]]), np.array([[1.0, -1.0]])),
        ]
        with self.subTest():
            for x0, x1 in datasets:
                with self.assertRaises(ValueError):
                    kern(x0, x1)
                with self.assertRaises(ValueError):
                    kern(x0)
