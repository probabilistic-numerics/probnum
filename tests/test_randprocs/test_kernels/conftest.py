"""Test fixtures for kernels."""

from typing import Callable, Optional

import numpy as np
import pytest

import probnum as pn
from probnum.typing import ShapeType


# Kernel objects
@pytest.fixture(
    params=[
        pytest.param(input_shape, id=f"inshape{input_shape}")
        for input_shape in [(), (1,), (10,), (100,)]
    ],
    name="input_shape",
)
def fixture_input_shape(request) -> ShapeType:
    """Input shape of the covariance function."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (pn.randprocs.kernels.Linear, {"constant": 1.0}),
            (pn.randprocs.kernels.WhiteNoise, {"sigma": -1.0}),
            (pn.randprocs.kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (pn.randprocs.kernels.ExpQuad, {"lengthscale": 1.5}),
            (pn.randprocs.kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 0.5, "nu": 0.5}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 1.5, "nu": 2.5}),
            # (pn.randprocs.kernels.Matern, {"lengthscale": 2.5, "nu": 7.0}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 3.0, "nu": float("inf")}),
        ]
    ],
    name="kernel",
    scope="package",
)
def fixture_kernel(request, input_shape: ShapeType) -> pn.randprocs.kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](input_shape=input_shape, **request.param[1])


@pytest.fixture(name="kernel_call_naive", scope="package")
def fixture_kernel_call_naive(
    kernel: pn.randprocs.kernels.Kernel,
) -> Callable[[pn.backend.ndarray, Optional[pn.backend.ndarray]], pn.backend.ndarray]:
    """Naive implementation of kernel broadcasting which applies the kernel function to
    scalar arguments while looping over the first dimensions of the inputs explicitly.

    Can be used as a reference implementation of `Kernel.__call__` vectorization.
    """

    if kernel.input_ndim == 0:
        kernel_vectorized = np.vectorize(kernel, signature="(),()->()")
    else:
        assert kernel.input_ndim == 1

        kernel_vectorized = np.vectorize(kernel, signature="(d),(d)->()")

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
    name="x0_batch_shape",
    scope="package",
)
def fixture_x0_batch_shape(request) -> ShapeType:
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
    name="x1_batch_shape",
    scope="package",
)
def fixture_x1_batch_shape(request) -> Optional[ShapeType]:
    """Batch shape of the second argument of ``Kernel.matrix`` or ``None`` if the second
    argument is ``None``."""
    return request.param


@pytest.fixture(name="x0", scope="package")
def fixture_x0(x0_batch_shape: ShapeType) -> pn.backend.ndarray:
    """Random data from a standard normal distribution."""
    seed = pn.backend.random.split(pn.backend.random.seed(abs(hash(x0_batch_shape))))[0]

    return pn.backend.random.standard_normal(seed, shape=x0_batch_shape)


@pytest.fixture(name="x1", scope="package")
def fixture_x1(x1_batch_shape: Optional[ShapeType]) -> Optional[pn.backend.ndarray]:
    """Random data from a standard normal distribution."""
    if x1_batch_shape is None:
        return None

    seed = pn.backend.random.split(pn.backend.random.seed(abs(hash(x1_shape))))[1]

    return pn.backend.random.standard_normal(seed, shape=x1_batch_shape + input_shape)
