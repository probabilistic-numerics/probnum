"""Test cases for the product Matern kernel."""

import functools
import operator

import pytest

from probnum import backend, compat
from probnum.backend.typing import ArrayLike, ShapeType
from probnum.randprocs import kernels
import tests.utils


@pytest.mark.parametrize("lengthscale", [1.25])
@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.0])
def test_kernel_matrix(input_shape: ShapeType, lengthscale: float, nu: float):
    """Check that the product Matérn kernel matrix is an elementwise product of 1D
    Matérn kernel matrices."""
    if len(input_shape) > 1:
        pytest.skip()

    matern = kernels.Matern(input_shape=(), lengthscale=lengthscale, nu=nu)
    product_matern = kernels.ProductMatern(
        input_shape=input_shape, lengthscales=lengthscale, nus=nu
    )

    xs_shape = (15,) + input_shape
    xs = backend.random.uniform(
        seed=tests.utils.random.seed_from_sampling_args(base_seed=42, shape=xs_shape),
        shape=xs_shape,
    )

    kernel_matrix1 = product_matern.matrix(xs)

    if len(input_shape) > 0:
        assert len(input_shape) == 1

        kernel_matrix2 = functools.reduce(
            operator.mul, (matern.matrix(xs[:, dim]) for dim in range(input_shape[0]))
        )
    else:
        kernel_matrix2 = matern.matrix(xs)

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
