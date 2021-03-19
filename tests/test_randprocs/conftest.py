"""Fixtures for random process tests."""

from typing import Callable

import numpy as np
import pytest

from probnum import kernels, randprocs, statespace


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(1)],
    name="random_state",
)
def fixture_random_state(request):
    """Random state(s) used for test parameterization."""
    return np.random.RandomState(seed=request.param)


@pytest.fixture(
    params=[
        pytest.param(input_dim, id=f"indim{input_dim}") for input_dim in [1, 10, 100]
    ],
    name="input_dim",
)
def fixture_input_dim(request) -> int:
    """Input dimension of the random process."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(output_dim, id=f"outdim{output_dim}") for output_dim in [1, 2, 10]
    ]
)
def output_dim(request) -> int:
    """Output dimension of the random process."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(mu, id=mu[0])
        for mu in [
            ("zero", lambda x: np.zeros(x.shape[0])),
            ("lin", lambda x: 2 * x.sum(axis=1) + 1.0),
        ]
    ],
    name="mean",
)
def fixture_mean(request) -> Callable:
    """Mean function of a random process."""
    return request.param[1]


@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (kernels.ExpQuad, {"lengthscale": 1.5}),
            (kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
        ]
    ],
    name="cov",
)
def fixture_cov(request, input_dim: int) -> kernels.Kernel:
    """Covariance function."""
    return request.param[0](**request.param[1], input_dim=input_dim)


@pytest.fixture(name="gaussian_process")
def fixture_gaussian_process(mean, cov) -> randprocs.GaussianProcess:
    """Gaussian process."""
    return randprocs.GaussianProcess(mean=mean, cov=cov)


@pytest.fixture(
    params=[
        pytest.param(gmpdef, id=gmpdef[0])
        for gmpdef in [("brownian", statespace.IBM(ordint=1, spatialdim=1), 0.0, 0.0)]
    ],
    name="gauss_markov_process",
)
def fixture_gauss_markov_process(request) -> randprocs.GaussMarkovProcess:
    """Gauss-Markov process."""
    return randprocs.GaussMarkovProcess(
        linear_sde=request.param[1], t0=request.param[2], x0=request.param[3]
    )
