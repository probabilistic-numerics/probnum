"""Test fixtures for kernels."""

from typing import Callable, Optional

import numpy as np
import pytest

import probnum as pn
from probnum.typing import ShapeType

from ._utils import _shape_param_to_id_str


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
        pytest.param(input_dim, id=f"indim{input_dim}") for input_dim in [1, 10, 100]
    ],
    name="input_dim",
)
def fixture_input_dim(request) -> int:
    """Input dimension of the covariance function."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(output_dim, id=f"outdim{output_dim}") for output_dim in [1, 2, 10]
    ]
)
def output_dim(request) -> int:
    """Output dimension of the covariance function."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (pn.kernels.Linear, {"constant": 1.0}),
            (pn.kernels.WhiteNoise, {"sigma": -1.0}),
            (pn.kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (pn.kernels.ExpQuad, {"lengthscale": 1.5}),
            (pn.kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (pn.kernels.Matern, {"lengthscale": 0.5, "nu": 0.5}),
            (pn.kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
            (pn.kernels.Matern, {"lengthscale": 1.5, "nu": 2.5}),
            (pn.kernels.Matern, {"lengthscale": 2.5, "nu": 7.0}),
            (pn.kernels.Matern, {"lengthscale": 3.0, "nu": np.inf}),
        ]
    ],
    name="kernel",
)
def fixture_kernel(request, input_dim: int) -> pn.kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](**request.param[1], input_dim=input_dim)


@pytest.fixture(name="kernel_call_naive")
def fixture_kernel_call_naive(
    kernel: pn.kernels.Kernel,
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """Naive implementation of kernel broadcasting which applies the kernel function to
    scalar arguments while looping over the first dimensions of the inputs explicitly.
    Can be used as a reference implementation of kernel broadcasting."""

    kernel_vectorized = np.vectorize(kernel, signature="(d),(d)->()")

    def _kernel_naive(
        x0: np.ndarray,
        x1: Optional[np.ndarray],
    ):
        x0, _ = np.broadcast_arrays(
            x0,
            # This broadcasts x0 to have `input_dim` elements along its last axis
            np.empty_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=(kernel.input_dim,),
            ),
        )

        if x1 is None:
            x1 = x0
        else:
            x1, _ = np.broadcast_arrays(
                x1,
                # This broadcasts x1 to have `input_dim` elements along its last axis
                np.empty_like(  # pylint: disable=unexpected-keyword-arg
                    x1,
                    shape=(kernel.input_dim,),
                ),
            )

        assert x0.shape[-1] == kernel.input_dim and x1.shape[-1] == kernel.input_dim

        return kernel_vectorized(x0, x1)

    return _kernel_naive


# Test data for `Kernel.matrix`
D_IN = None


@pytest.fixture(
    params=[
        pytest.param(shape_param, id=f"x0{_shape_param_to_id_str(shape_param)}")
        for shape_param in [
            (1, 1),
            (1, D_IN),
            (2, 1),
            (2, D_IN),
            (10, 1),
            (10, D_IN),
            (100, 1),
            (100, D_IN),
        ]
    ],
    name="x0_shape",
)
def fixture_x0_shape(request, input_dim: int) -> ShapeType:
    """Shape of the first argument of ``Kernel.matrix``."""
    return tuple(input_dim if dim is D_IN else dim for dim in request.param)


@pytest.fixture(
    params=[
        pytest.param(shape_param, id=f"x1{_shape_param_to_id_str(shape_param)}")
        for shape_param in [
            None,
            (1, 1),
            (1, D_IN),
            (2, 1),
            (2, D_IN),
            (10, 1),
            (10, D_IN),
        ]
    ],
    name="x1_shape",
)
def fixture_x1_shape(request, input_dim: int) -> ShapeType:
    """Shape of the second argument of ``Kernel.matrix`` or ``None`` if the second
    argument is ``None``."""
    if request.param is None:
        return None

    return tuple(input_dim if dim is D_IN else dim for dim in request.param)


@pytest.fixture(name="x0")
def fixture_x0(rng: np.random.Generator, x0_shape: ShapeType) -> np.ndarray:
    """Random data from a standard normal distribution."""
    return rng.normal(0, 1, size=x0_shape)


@pytest.fixture(name="x1")
def fixture_x1(
    rng: np.random.Generator, x1_shape: Optional[ShapeType]
) -> Optional[np.ndarray]:
    """Random data from a standard normal distribution."""
    if x1_shape is None:
        return None

    return rng.normal(0, 1, size=x1_shape)
