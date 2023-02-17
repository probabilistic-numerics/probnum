"""Test cases for the product Matern covariance function."""

import numpy as np
import pytest

from probnum.randprocs import covfuncs
import probnum.utils as _utils


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.0])
def test_covfun_matrix(input_dim, nu):
    """Check that the product Matérn covariance function matrix is an elementwise
    product of 1D Matérn covariance function matrices."""
    lengthscale = 1.25
    matern = covfuncs.Matern(input_shape=(1,), lengthscales=lengthscale, nu=nu)
    product_matern = covfuncs.ProductMatern(
        input_shape=(input_dim,), lengthscales=lengthscale, nus=nu
    )
    rng = np.random.default_rng(42)
    num_xs = 15
    xs = rng.random(size=(num_xs, input_dim))
    k_matrix1 = product_matern.matrix(xs)
    k_matrix2 = np.ones(shape=(num_xs, num_xs))
    for dim in range(input_dim):
        k_matrix2 *= matern.matrix(_utils.as_colvec(xs[:, dim]))
    np.testing.assert_allclose(
        k_matrix1,
        k_matrix2,
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
    """Parameters must be scalars if covariance function input is scalar."""
    with pytest.raises(ValueError):
        covfuncs.ProductMatern(input_shape=(), lengthscales=ell, nus=nu)
