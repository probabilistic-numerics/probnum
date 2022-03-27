"""Fixtures for random variable arithmetic."""
import pytest

from probnum import backend, linops, randvars
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.typing import ShapeLike
import tests.utils


@pytest.fixture
def constant(shape_const: ShapeLike) -> randvars.Constant:
    seed = tests.utils.random.seed_from_sampling_args(
        base_seed=19836, shape=shape_const
    )

    return randvars.Constant(
        support=backend.random.standard_normal(seed, shape=shape_const)
    )


@pytest.fixture
def multivariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    seed = tests.utils.random.seed_from_sampling_args(base_seed=1908, shape=shape)
    seed_mean, seed_cov = backend.random.split(seed)

    rv = randvars.Normal(
        mean=backend.random.standard_normal(seed_mean, shape=shape),
        cov=random_spd_matrix(seed_cov, dim=shape[0]),
    )
    if precompute_cov_cholesky:
        rv._compute_cov_cholesky()
    return rv


@pytest.fixture
def matrixvariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    seed = tests.utils.random.seed_from_sampling_args(base_seed=354, shape=shape)
    seed_mean, seed_cov_A, seed_cov_B = backend.random.split(seed, num=3)

    rv = randvars.Normal(
        mean=backend.random.standard_normal(seed_mean, shape=shape),
        cov=linops.Kronecker(
            A=random_spd_matrix(seed_cov_A, dim=shape[0]),
            B=random_spd_matrix(seed_cov_B, dim=shape[1]),
        ),
    )
    if precompute_cov_cholesky:
        rv._compute_cov_cholesky()
    return rv


@pytest.fixture
def symmetric_matrixvariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    seed = tests.utils.random.seed_from_sampling_args(base_seed=246, shape=shape)
    seed_mean, seed_cov = backend.random.split(seed)

    rv = randvars.Normal(
        mean=random_spd_matrix(seed_mean, dim=shape[0]),
        cov=linops.SymmetricKronecker(A=random_spd_matrix(seed_cov, dim=shape[0])),
    )
    if precompute_cov_cholesky:
        rv._compute_cov_cholesky()
    return rv
