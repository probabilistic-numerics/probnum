"""Test cases for the Matern kernel."""

import numpy as np

import probnum.kernels as kerns

from .test_kernel import KernelTestCase


class MaternTestCase(KernelTestCase):
    """Test case for Matern kernels."""

    def test_nonpositive_nu_raises_exception(self):
        """Check whether a non-positive nu parameter raises a ValueError."""
        for nu in [-1, -1.0, 0.0, 0]:
            with self.subTest():
                with self.assertRaises(ValueError):
                    kerns.Matern(input_dim=1, nu=nu)

    def test_nu_large_recovers_rbf_kernel(self):
        """Test whether a Matern kernel with nu large is close to an RBF kernel."""
        x0 = self.data_nxd_0
        x1 = self.data_nxd_1
        lengthscale = 1.25
        kernmat_rbf = kerns.ExpQuad(lengthscale=lengthscale, input_dim=x0.shape[1])
        kernmat_matern = kerns.Matern(
            lengthscale=lengthscale, nu=100, input_dim=x0.shape[1]
        )
        self.assertAllClose(
            kernmat_rbf(x0, x1),
            kernmat_matern(x0, x1),
            msg="RBF and Matern kernel are not equivalent for nu=infty.",
            rtol=0.05,
            atol=0.01,
        )
