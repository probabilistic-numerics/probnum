"""Test cases defining linear systems to be solved."""

import numpy as np
from pytest_cases import case

from probnum import problems
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@case(tags=["spd"])
def case_random_spd_linsys(
    ncols: int,
) -> problems.LinearSystem:
    rng = np.random.default_rng(1)
    A = random_spd_matrix(rng=rng, dim=ncols)
    x = rng.normal(size=(ncols,))
    b = A @ x
    return problems.LinearSystem(A=A, b=b, solution=x)


@case(tags=["spd", "sparse"])
def case_random_sparse_spd_linsys(
    ncols: int,
) -> problems.LinearSystem:
    rng = np.random.default_rng(1)
    A = random_sparse_spd_matrix(rng=rng, dim=ncols, density=0.1)
    x = rng.normal(size=(ncols,))
    b = A @ x
    return problems.LinearSystem(A=A, b=b, solution=x)
