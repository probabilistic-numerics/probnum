"""Test cases defining random variables with a normal distribution."""

from pytest_cases import case, parametrize

from probnum import backend, randvars
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.typing import ScalarLike


@case(tags=["univariate"])
@parametrize("mean", (-1.0, 1))
@parametrize("var", (3.0, 2))
def case_univariate(mean: ScalarLike, var: ScalarLike) -> randvars.Normal:
    return randvars.Normal(mean, var)


@case(tags=["vectorvariate"])
@parametrize("dim", [1, 2, 5, 10, 20])
def case_vectorvariate(dim: int) -> randvars.Normal:
    mean = backend.random.standard_normal(backend.random.seed(654 + dim), shape=(dim,))
    cov = random_spd_matrix(backend.random.seed(846), dim)

    return randvars.Normal(mean, cov)
