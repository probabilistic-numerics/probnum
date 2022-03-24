"""Test cases for the product Matern kernel."""

import numpy as np
import pytest

from probnum.randprocs import kernels


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.0])
def test_kernel_matrix(input_dim, nu):
    """Check that the product Matérn kernel matrix is an elementwise product of 1D
    Matérn kernel matrices."""
    lengthscale = 1.25
    matern = kernels.Matern(input_shape=(1,), lengthscale=lengthscale, nu=nu)
    product_matern = kernels.ProductMatern(
        input_shape=(input_dim,), lengthscales=lengthscale, nus=nu
    )
    rng = np.random.default_rng(42)
    num_xs = 15
    xs = rng.random(size=(num_xs, input_dim))
    kernel_matrix1 = product_matern.matrix(xs)
    kernel_matrix2 = np.ones(shape=(num_xs, num_xs))
    for dim in range(input_dim):
        kernel_matrix2 *= matern.matrix(xs[:, [dim]])
    np.testing.assert_allclose(
        kernel_matrix1,
        kernel_matrix2,
    )


@pytest.mark.parametrize(
    "ell,nu",
    [
        (np.array([3.0]), 0.5),
        (3.0, np.array([0.5])),
        (np.array([3.0]), np.array([0.5])),
    ],
)
def test_wrong_initialization_raises_exception(ell, nu):
    """Parameters must be scalars if kernel input is scalar."""
    with pytest.raises(ValueError):
        kernels.ProductMatern(input_shape=(), lengthscales=ell, nus=nu)
