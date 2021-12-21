"""Fixtures for random variable arithmetic."""
import numpy as np
import pytest

from probnum import backend, linops, randvars
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.typing import ShapeLike
from tests.testing import seed_from_args


@pytest.fixture
def constant(shape_const: ShapeLike) -> randvars.Constant:
    seed = seed_from_args(shape_const, 19836)

    return randvars.Constant(
        support=backend.random.standard_normal(seed, shape=shape_const)
    )


@pytest.fixture
def multivariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    seed = seed_from_args(shape, precompute_cov_cholesky, 1908)
    seed_mean, seed_cov = backend.random.split(seed)

    rv = randvars.Normal(
        mean=backend.random.standard_normal(seed_mean, shape=shape),
        cov=random_spd_matrix(seed_cov, dim=shape[0]),
    )
    if precompute_cov_cholesky:
        rv.compute_cov_cholesky()
    return rv


@pytest.fixture
def matrixvariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    seed = seed_from_args(shape, precompute_cov_cholesky, 354)
    seed_mean, seed_cov_A, seed_cov_B = backend.random.split(seed, num=3)

    rv = randvars.Normal(
        mean=backend.random.standard_normal(seed_mean, shape=shape),
        cov=linops.Kronecker(
            A=random_spd_matrix(seed_cov_A, dim=shape[0]),
            B=random_spd_matrix(seed_cov_B, dim=shape[1]),
        ),
    )
    if precompute_cov_cholesky:
        rv.compute_cov_cholesky()
    return rv


@pytest.fixture
def symmetric_matrixvariate_normal(
    shape: ShapeLike, precompute_cov_cholesky: bool
) -> randvars.Normal:
    seed = seed_from_args(shape, precompute_cov_cholesky, 246)
    seed_mean, seed_cov = backend.random.split(seed)

    rv = randvars.Normal(
        mean=random_spd_matrix(seed_mean, dim=shape[0]),
        cov=linops.SymmetricKronecker(A=random_spd_matrix(seed_cov, dim=shape[0])),
    )
    if precompute_cov_cholesky:
        rv.compute_cov_cholesky()
    return rv
