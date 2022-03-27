"""Test fixtures for kernels."""

from typing import Callable, Optional

import pytest

from probnum import backend
from probnum.randprocs import kernels
from probnum.typing import ArrayType, ShapeType
import tests.utils


# Kernel objects
@pytest.fixture(
    params=[
        pytest.param(input_shape, id=f"inshape{input_shape}")
        for input_shape in [(), (1,), (10,), (100,)]
    ],
    scope="package",
)
def input_shape(request) -> ShapeType:
    """Input shape of the covariance function."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (kernels.Linear, {"constant": 1.0}),
            (kernels.WhiteNoise, {"sigma_sq": 1.0}),
            (kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (kernels.ExpQuad, {"lengthscale": 1.5}),
            (kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (kernels.Matern, {"lengthscale": 0.5, "nu": 0.5}),
            (kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
            (kernels.Matern, {"lengthscale": 1.5, "nu": 2.5}),
            (kernels.Matern, {"lengthscale": 2.5, "nu": 7.0}),
            (kernels.Matern, {"lengthscale": 3.0, "nu": backend.inf}),
            (kernels.ProductMatern, {"lengthscales": 0.5, "nus": 0.5}),
        ]
    ],
    scope="package",
)
def kernel(request, input_shape: ShapeType) -> kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](input_shape=input_shape, **request.param[1])


@pytest.mark.skipif_backend(backend.Backend.TORCH)
@pytest.fixture(scope="package")
def kernel_call_naive(
    kernel: kernels.Kernel,
) -> Callable[[ArrayType, Optional[ArrayType]], ArrayType]:
    """Naive implementation of kernel broadcasting which applies the kernel function to
    scalar arguments while looping over the first dimensions of the inputs explicitly.

    Can be used as a reference implementation of `Kernel.__call__` vectorization.
    """

    if kernel.input_ndim == 0:
        kernel_vectorized = backend.vectorize(kernel, signature="(),()->()")
    else:
        assert kernel.input_ndim == 1

        kernel_vectorized = backend.vectorize(kernel, signature="(d),(d)->()")

    return lambda x0, x1: (
        kernel_vectorized(x0, x0) if x1 is None else kernel_vectorized(x0, x1)
    )


# Test data for `Kernel.matrix`
@pytest.fixture(
    params=[
        pytest.param(shape, id=f"x0{shape}")
        for shape in [
            (),
            (1,),
            (2,),
            (10,),
            (100,),
        ]
    ],
    scope="package",
)
def x0_batch_shape(request) -> ShapeType:
    """Batch shape of the first argument of ``Kernel.matrix``."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(shape, id=f"x1{shape}")
        for shape in [
            None,
            (),
            (1,),
            (3,),
            (10,),
        ]
    ],
    scope="package",
)
def x1_batch_shape(request) -> Optional[ShapeType]:
    """Batch shape of the second argument of ``Kernel.matrix`` or ``None`` if the second
    argument is ``None``."""
    return request.param


@pytest.fixture(scope="package")
def x0(input_shape: ShapeType, x0_batch_shape: ShapeType) -> ArrayType:
    """Random data from a standard normal distribution."""
    shape = x0_batch_shape + input_shape

    seed = tests.utils.random.seed_from_sampling_args(base_seed=34897, shape=shape)

    return backend.random.standard_normal(seed, shape=shape)


@pytest.fixture(scope="package")
def x1(input_shape: ShapeType, x1_batch_shape: ShapeType) -> Optional[ArrayType]:
    """Random data from a standard normal distribution."""
    if x1_batch_shape is None:
        return None

    shape = x1_batch_shape + input_shape

    seed = tests.utils.random.seed_from_sampling_args(base_seed=533, shape=shape)

    return backend.random.standard_normal(seed, shape=shape)
