"""Test fixtures for kernels."""

from typing import Optional

import numpy as np
import pytest

import probnum.kernels as kernels


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(1)],
    name="rng",
)
def fixture_rng(request):
    """Random state(s) used for test parameterization."""
    return np.random.default_rng(seed=request.param)


@pytest.fixture(
    params=[
        pytest.param(num_data, id=f"ndata{num_data}") for num_data in [1, 2, 10, 100]
    ],
    name="num_data",
)
def fixture_num_data(request) -> int:
    """Size of the dataset."""
    return request.param


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


# Datasets
@pytest.fixture(name="x0")
def fixture_x0(num_data: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random data from a standard normal distribution."""
    return rng.normal(0, 1, size=(num_data, input_dim))


@pytest.fixture(
    params=[
        pytest.param(num_data, id=f"ndata{num_data}") for num_data in [None, 10, 2, 1]
    ],
    name="x1",
)
def fixture_x1(
    request, input_dim: int, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Random data from a standard normal distribution."""
    if request.param is None:
        return None
    else:
        return rng.normal(0, 1, size=(request.param, input_dim))


@pytest.fixture()
def x0_1d(input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random 1D dataset."""
    return rng.normal(0, 1, size=(input_dim,))


# Kernel and kernel matrices


@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (kernels.Linear, {"constant": 1.0}),
            (kernels.WhiteNoise, {"sigma": -1.0}),
            (kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (kernels.ExpQuad, {"lengthscale": 1.5}),
            (kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (kernels.Matern, {"lengthscale": 0.5, "nu": 0.5}),
            (kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
            (kernels.Matern, {"lengthscale": 1.5, "nu": 2.5}),
            (kernels.Matern, {"lengthscale": 2.5, "nu": 7.0}),
            (kernels.Matern, {"lengthscale": 3.0, "nu": np.inf}),
        ]
    ],
    name="kernel",
)
def fixture_kernel(request, input_dim: int) -> kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](**request.param[1], input_dim=input_dim)


@pytest.fixture()
def kernmat(
    kernel: kernels.Kernel, x0: np.ndarray, x1: Optional[np.ndarray]
) -> np.ndarray:
    """Kernel evaluated at the data."""
    return kernel(x0=x0, x1=x1)
