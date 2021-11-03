"""Fixtures for random variable arithmetic."""
import numpy as np
import pytest

from probnum import linops, randvars
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.typing import ShapeArgType


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def constant(shape_const: ShapeArgType, rng: np.random.Generator) -> randvars.Constant:
    return randvars.Constant(support=rng.normal(size=shape_const))


@pytest.fixture
def multivariate_normal(
    shape: ShapeArgType, precompute_cov_cholesky: bool, rng: np.random.Generator
) -> randvars.Normal:
    rv = randvars.Normal(
        mean=rng.normal(size=shape),
        cov=random_spd_matrix(rng=rng, dim=shape[0]),
    )
    if precompute_cov_cholesky:
        rv.precompute_cov_cholesky()
    return rv


@pytest.fixture
def matrixvariate_normal(
    shape: ShapeArgType, precompute_cov_cholesky: bool, rng: np.random.Generator
) -> randvars.Normal:
    rv = randvars.Normal(
        mean=rng.normal(size=shape),
        cov=linops.Kronecker(
            A=random_spd_matrix(dim=shape[0], rng=rng),
            B=random_spd_matrix(dim=shape[1], rng=rng),
        ),
    )
    if precompute_cov_cholesky:
        rv.precompute_cov_cholesky()
    return rv


@pytest.fixture
def symmetric_matrixvariate_normal(
    shape: ShapeArgType, precompute_cov_cholesky: bool, rng: np.random.Generator
) -> randvars.Normal:
    rv = randvars.Normal(
        mean=random_spd_matrix(dim=shape[0], rng=rng),
        cov=linops.SymmetricKronecker(A=random_spd_matrix(dim=shape[0], rng=rng)),
    )
    if precompute_cov_cholesky:
        rv.precompute_cov_cholesky()
    return rv
