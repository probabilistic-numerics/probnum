"""Fixtures for Markov processes."""

import numpy as np

from probnum import backend, randvars
from probnum.problems.zoo.linalg import random_spd_matrix

import pytest
from tests.utils.random import rng_state_from_sampling_args


@pytest.fixture(params=[2])
def state_dim(request) -> int:
    """State dimension."""
    return request.param


# Covariance matrices


@pytest.fixture
def spdmat1(state_dim: int):
    rng_state = rng_state_from_sampling_args(base_seed=3245956, shape=state_dim)
    return random_spd_matrix(rng_state, shape=(state_dim, state_dim))


@pytest.fixture
def spdmat2(state_dim: int):
    rng_state = rng_state_from_sampling_args(base_seed=1, shape=state_dim)
    return random_spd_matrix(rng_state, shape=(state_dim, state_dim))


@pytest.fixture
def spdmat3(state_dim: int):
    rng_state = rng_state_from_sampling_args(base_seed=2498, shape=state_dim)
    return random_spd_matrix(rng_state, shape=(state_dim, state_dim))


@pytest.fixture
def spdmat4(state_dim: int):
    rng_state = rng_state_from_sampling_args(base_seed=4056, shape=state_dim)
    return random_spd_matrix(rng_state, shape=(state_dim, state_dim))


# 'Normal' random variables


@pytest.fixture
def some_normal_rv1(state_dim, spdmat1):
    rng_state = rng_state_from_sampling_args(base_seed=6879, shape=spdmat1.shape)
    return randvars.Normal(
        mean=backend.random.uniform(rng_state=rng_state, shape=state_dim),
        cov=spdmat1,
        cache={"cov_cholesky": np.linalg.cholesky(spdmat1)},
    )


@pytest.fixture
def some_normal_rv2(state_dim, spdmat2):
    rng_state = rng_state_from_sampling_args(base_seed=2344, shape=spdmat2.shape)
    return randvars.Normal(
        mean=backend.random.uniform(rng_state=rng_state, shape=state_dim),
        cov=spdmat2,
        cache={"cov_cholesky": np.linalg.cholesky(spdmat2)},
    )


@pytest.fixture
def some_normal_rv3(state_dim, spdmat3):
    rng_state = rng_state_from_sampling_args(base_seed=76, shape=spdmat3.shape)
    return randvars.Normal(
        mean=backend.random.uniform(rng_state=rng_state, shape=state_dim),
        cov=spdmat3,
        cache={"cov_cholesky": np.linalg.cholesky(spdmat3)},
    )


@pytest.fixture
def some_normal_rv4(state_dim, spdmat4):
    rng_state = rng_state_from_sampling_args(base_seed=22, shape=spdmat4.shape)
    return randvars.Normal(
        mean=backend.random.uniform(rng_state=rng_state, shape=state_dim),
        cov=spdmat4,
        cache={"cov_cholesky": np.linalg.cholesky(spdmat4)},
    )


@pytest.fixture
def diffusion():
    """A diffusion != 1 makes it easier to see if _diffusion is actually used in forward
    and backward."""
    return 5.1412512431


@pytest.fixture(params=["classic", "sqrt"])
def forw_impl_string_linear_gauss(request):
    """Forward implementation choices passed via strings."""
    return request.param


@pytest.fixture(params=["classic", "joseph", "sqrt"])
def backw_impl_string_linear_gauss(request):
    """Backward implementation choices passed via strings."""
    return request.param
