"""Fixtures for random process tests."""

import functools
from typing import Callable

import numpy as np
import pytest

from probnum import LambdaFunction, randprocs
from probnum.randprocs import kernels, mean_fns


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(3)],
    name="rng",
)
def fixture_rng(request) -> np.random.Generator:
    """Random state(s) used for test parameterization."""
    return np.random.default_rng(seed=request.param)


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
            ("zero", mean_fns.Zero),
            (
                "lin",
                functools.partial(LambdaFunction, lambda x: 2 * x.sum(axis=1) + 1.0),
            ),
        ]
    ],
    name="mean",
)
def fixture_mean(request, input_dim: int) -> Callable:
    """Mean function of a random process."""
    return request.param[1](input_shape=(input_dim,), output_shape=())


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
    return request.param[0](**request.param[1], input_shape=(input_dim,))


@pytest.fixture(
    params=[
        pytest.param(randprocdef, id=randprocdef[0])
        for randprocdef in [
            (
                "gp",
                randprocs.GaussianProcess(
                    mean=mean_fns.Zero(input_shape=(1,)),
                    cov=kernels.Matern(input_shape=(1,)),
                ),
            ),
        ]
    ],
    name="random_process",
)
def fixture_random_process(request) -> randprocs.RandomProcess:
    """Random process."""
    return request.param[1]


@pytest.fixture(name="gaussian_process")
def fixture_gaussian_process(mean, cov) -> randprocs.GaussianProcess:
    """Gaussian process."""
    return randprocs.GaussianProcess(mean=mean, cov=cov)


@pytest.fixture(params=[pytest.param(n, id=f"n{n}") for n in [1, 10]], name="args0")
def fixture_args0(
    request,
    random_process: randprocs.RandomProcess,
    rng: np.random.Generator,
) -> np.ndarray:
    """Input(s) to a random process."""
    return rng.normal(size=(request.param,) + random_process.input_shape)
