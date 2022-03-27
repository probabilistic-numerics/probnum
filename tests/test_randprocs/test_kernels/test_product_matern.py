"""Test cases for the product Matern kernel."""

import functools
import operator

import pytest

from probnum import backend, compat
from probnum.randprocs import kernels
from probnum.typing import ArrayLike
from tests import testing


@pytest.mark.parametrize("lengthscale", [1.25])
@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.0])
def test_kernel_matrix(input_dim: int, lengthscale: float, nu: float):
    """Check that the product Matérn kernel matrix is an elementwise product of 1D
    Matérn kernel matrices."""
    matern = kernels.Matern(input_shape=(1,), lengthscale=lengthscale, nu=nu)
    product_matern = kernels.ProductMatern(
        input_shape=(input_dim,), lengthscales=lengthscale, nus=nu
    )

    num_xs = 15
    xs_shape = (num_xs, input_dim)
    xs = backend.random.uniform(
        seed=testing.seed_from_sampling_args(base_seed=42, shape=xs_shape),
        shape=xs_shape,
    )

    kernel_matrix1 = product_matern.matrix(xs)
    kernel_matrix2 = functools.reduce(
        operator.mul, (matern.matrix(xs[:, [dim]]) for dim in range(input_dim))
    )

    compat.testing.assert_allclose(kernel_matrix1, kernel_matrix2)


@pytest.mark.parametrize(
    "ell,nu",
    [
        ([3.0], 0.5),
        (3.0, [0.5]),
        ([3.0], [0.5]),
    ],
)
def test_wrong_initialization_raises_exception(ell: ArrayLike, nu: ArrayLike):
    """Parameters must be scalars if kernel input is scalar."""
    with pytest.raises(ValueError):
        kernels.ProductMatern(input_shape=(), lengthscales=ell, nus=nu)
