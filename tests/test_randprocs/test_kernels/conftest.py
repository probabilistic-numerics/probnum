"""Test fixtures for kernels."""

from typing import Callable, Optional

import numpy as np
import pytest

import probnum as pn
from probnum.typing import ShapeType


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(1)],
    name="rng",
)
def fixture_rng(request):
    """Random state(s) used for test parameterization."""
    return np.random.default_rng(seed=request.param)


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
            (pn.randprocs.kernels.WhiteNoise, {"sigma_sq": 1.0}),
            (pn.randprocs.kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (pn.randprocs.kernels.ExpQuad, {"lengthscale": 1.5}),
            (pn.randprocs.kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 0.5, "nu": 0.5}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 1.5, "nu": 2.5}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 2.5, "nu": 7.0}),
            (pn.randprocs.kernels.Matern, {"lengthscale": 3.0, "nu": np.inf}),
            (pn.randprocs.kernels.ProductMatern, {"lengthscales": 0.5, "nus": 0.5}),
        ]
    ],
    name="kernel",
)
def fixture_kernel(request, input_shape: ShapeType) -> pn.randprocs.kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](input_shape=input_shape, **request.param[1])


@pytest.fixture(name="kernel_call_naive")
def fixture_kernel_call_naive(
    kernel: pn.randprocs.kernels.Kernel,
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
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
)
def fixture_x1_batch_shape(request) -> Optional[ShapeType]:
    """Batch shape of the second argument of ``Kernel.matrix`` or ``None`` if the second
    argument is ``None``."""
    return request.param


@pytest.fixture(name="x0")
def fixture_x0(
    rng: np.random.Generator, x0_batch_shape: ShapeType, input_shape: ShapeType
) -> np.ndarray:
    """Random data from a standard normal distribution."""
    return rng.normal(0, 1, size=x0_batch_shape + input_shape)


@pytest.fixture(name="x1")
def fixture_x1(
    rng: np.random.Generator,
    x1_batch_shape: Optional[ShapeType],
    input_shape: ShapeType,
) -> Optional[np.ndarray]:
    """Random data from a standard normal distribution."""
    if x1_batch_shape is None:
        return None

    return rng.normal(0, 1, size=x1_batch_shape + input_shape)
