"""Test fixtures for kernels."""

from typing import Optional, Union

import numpy as np
import pytest

import probnum.kernels as kernels
import probnum.quad._integration_measures as measures
import probnum.quad._kernel_embeddings as kernel_embeddings
from probnum.type import FloatArgType


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(1)],
    name="random_state",
)
def fixture_random_state(request):
    """Random state(s) used for test parameterization."""
    return np.random.RandomState(seed=request.param)


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
        pytest.param(input_dim, id=f"dim{input_dim}") for input_dim in [1, 10, 100]
    ],
    name="input_dim",
)
def fixture_input_dim(request) -> int:
    """Input dimension of the covariance function."""
    return request.param


# Datasets
@pytest.fixture(name="x")
def fixture_x(
    input_dim: int, num_data: int, random_state: np.random.RandomState
) -> np.ndarray:
    """Random data from a standard normal distribution."""
    return random_state.normal(0, 1, size=(input_dim, num_data))


# Kernels
@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (kernels.ExpQuad, {"lengthscale": 1.5}),
        ]
    ],
    name="kernel",
)
def fixture_kernel(request, input_dim: int) -> kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](**request.param[1], input_dim=input_dim)


# Measures
@pytest.fixture(name="mean")
def fixture_mean(input_dim: int, random_state: np.random.RandomState) -> np.ndarray:
    """Random array for mean from a standard normal distribution."""
    if input_dim == 1:
        return random_state.normal(0, 1)
    return random_state.normal(0, 1, size=(input_dim, 1))


@pytest.fixture(name="cov")
def fixture_cov(
    input_dim: int, diagonal, random_state: np.random.RandomState
) -> np.ndarray:
    """Random covariance matrix."""
    if input_dim == 1:
        return random_state.uniform(0.5, 1.5)
    if diagonal:
        return random_state.uniform(0.5, 1.5, size=(input_dim, 1))
    mat = random_state.normal(0, 1, size=(input_dim, input_dim))
    return mat @ mat.T


@pytest.fixture(
    params=[
        pytest.param(diagonal, id=f"covdiag{diagonal}") for diagonal in [True, False]
    ],
    name="diagonal",
)
def fixture_diagonal(request) -> str:
    """Possible forms of the covariance matrix."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(mdef, id=mdef[0].__name__)
        for mdef in [
            (measures.GaussianMeasure,),
        ]
    ],
    name="measure",
)
def fixture_measure(
    request,
    mean: Optional[Union[np.ndarray, FloatArgType]],
    cov: Optional[Union[np.ndarray, FloatArgType]],
) -> measures.IntegrationMeasure:
    """Kernel / covariance function."""
    return request.param[0](mean=mean, cov=cov)


# Kernel Embeddings
@pytest.fixture(
    params=[
        pytest.param(kedef, id=kedef[0].__name__)
        for kedef in [
            (kernel_embeddings._KExpQuadMGauss,),
        ]
    ],
    name="kernel_embedding",
)
def fixture_kernel_embedding(
    request, kernel: kernels.Kernel, measure: measures.IntegrationMeasure
) -> kernel_embeddings._KernelEmbedding:
    """Kernel / covariance function."""
    return request.param[0](kernel=kernel, measure=measure)
