"""Test fixtures for kernels."""

from typing import Dict

import numpy as np
import pytest

import probnum.kernels as kernels
import probnum.quad._integration_measures as measures
from probnum.quad.kernel_embeddings._kernel_embedding import KernelEmbedding

# pylint: disable=unnecessary-lambda


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(1)],
    name="rng",
)
def fixture_rng(request):
    """Random state(s) used for test parameterization."""
    return np.random.default_rng(seed=request.param)


@pytest.fixture(
    params=[pytest.param(num_data, id=f"ndata{num_data}") for num_data in [1, 2, 20]],
    name="num_data",
)
def fixture_num_data(request) -> int:
    """Size of the dataset."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(input_dim, id=f"dim{input_dim}") for input_dim in [1, 2, 3, 5]
    ],
    name="input_dim",
)
def fixture_input_dim(request) -> int:
    """Input dimension of the covariance function."""
    return request.param


# Datasets
@pytest.fixture(name="x")
def fixture_x(input_dim: int, num_data: int, rng: np.random.Generator) -> np.ndarray:
    """Random data from a standard normal distribution."""
    return rng.normal(0, 1, size=(num_data, input_dim))


# Measures
@pytest.fixture(
    params=[
        pytest.param(diagonal, id="covdiag" if diagonal else "covfull")
        for diagonal in [True, False]
    ],
    name="cov_diagonal",
)
def fixture_diagonal(request) -> str:
    """Possible forms of the covariance matrix."""
    return request.param


@pytest.fixture(
    params=[pytest.param(name, id=name) for name in ["gauss", "lebesgue"]],
    name="measure_name",
)
def fixture_measure_names(request) -> str:
    """Pedestrian way to deal with integration measures."""
    return request.param


@pytest.fixture(name="measure_params")
def fixture_measure_params(
    measure_name: str,
    input_dim: int,
    cov_diagonal: bool,
    rng: np.random.Generator,
) -> Dict:
    params = {"name": measure_name}

    if measure_name == "gauss":
        # set up mean and covariance
        if input_dim == 1:
            mean = rng.normal(0, 1)
            cov = rng.uniform(0.5, 1.5)
        else:
            mean = rng.normal(0, 1, size=(input_dim, 1))
            if cov_diagonal:
                cov = rng.uniform(0.5, 1.5, size=(input_dim, 1))
            else:
                mat = rng.normal(0, 1, size=(input_dim, input_dim))
                cov = mat @ mat.T

        params["mean"] = mean
        params["cov"] = cov

    elif measure_name == "lebesgue":
        # set up bounds
        rv = rng.uniform(0, 1, size=(input_dim, 2))
        domain = (rv[:, 0] - 1.0, rv[:, 1] + 1.0)

        params["domain"] = domain
        params["normalized"] = True

    return params


@pytest.fixture(name="measure")
def fixture_measure(measure_params) -> measures.IntegrationMeasure:
    """Kernel / covariance function."""
    name = measure_params.pop("name")

    if name == "gauss":
        return measures.GaussianMeasure(**measure_params)
    elif name == "lebesgue":
        return measures.LebesgueMeasure(**measure_params)
    raise NotImplementedError


# Kernels
@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (kernels.ExpQuad, {"lengthscale": 1.25}),
        ]
    ],
    name="kernel",
)
def fixture_kernel(request, input_dim: int) -> kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](**request.param[1], input_dim=input_dim)


# Kernel Embeddings
@pytest.fixture(name="kernel_embedding")
def fixture_kernel_embedding(
    request, kernel: kernels.Kernel, measure: measures.IntegrationMeasure
) -> KernelEmbedding:
    """Set up kernel embedding."""
    return KernelEmbedding(kernel, measure)


# Test functions
@pytest.fixture(
    params=[
        pytest.param(fun, id=f"f={key}")
        for key, fun in {
            "x": lambda x: x,
            "x**2": lambda x: x ** 2,
            "sin(x)": lambda x: np.sin(x),
        }.items()
    ],
    name="f1d",
)
def fixture_f1d(request):
    """1D test function for BQ."""
    return request.param
