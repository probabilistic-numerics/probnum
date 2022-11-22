"""Fixtures for random variable arithmetic."""
from probnum import backend, linops, randvars
from probnum.backend.typing import ShapeLike
from probnum.problems.zoo.linalg import random_spd_matrix

import pytest
import tests.utils


@pytest.fixture
def constant(shape_const: ShapeLike) -> randvars.Constant:
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=19836, shape=shape_const
    )

    return randvars.Constant(
        support=backend.random.standard_normal(rng_state, shape=shape_const)
    )


@pytest.fixture
def multivariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=1908, shape=shape
    )
    rng_state_mean, rng_state_cov = backend.random.split(rng_state)

    rv = randvars.Normal(
        mean=backend.random.standard_normal(rng_state_mean, shape=shape),
        cov=random_spd_matrix(rng_state_cov, dim=shape[0]),
    )
    if precompute_cov_cholesky:
        rv._compute_cov_cholesky()
    return rv


@pytest.fixture
def matrixvariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=354, shape=shape
    )
    rng_state_mean, rng_state_cov_A, rng_state_cov_B = backend.random.split(
        rng_state, num=3
    )

    rv = randvars.Normal(
        mean=backend.random.standard_normal(rng_state_mean, shape=shape),
        cov=linops.Kronecker(
            A=random_spd_matrix(rng_state_cov_A, dim=shape[0]),
            B=random_spd_matrix(rng_state_cov_B, dim=shape[1]),
        ),
    )
    if precompute_cov_cholesky:
        rv._compute_cov_cholesky()
    return rv


@pytest.fixture
def symmetric_matrixvariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=246, shape=shape
    )
    rng_state_mean, rng_state_cov = backend.random.split(rng_state)

    rv = randvars.Normal(
        mean=random_spd_matrix(rng_state_mean, dim=shape[0]),
        cov=linops.SymmetricKronecker(A=random_spd_matrix(rng_state_cov, dim=shape[0])),
    )
    if precompute_cov_cholesky:
        rv._compute_cov_cholesky()
    return rv
