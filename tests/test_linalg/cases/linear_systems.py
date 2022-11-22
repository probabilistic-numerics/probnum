"""Linear system test cases."""

from typing import Union

import numpy as np
import scipy.sparse

from probnum import backend, linops, problems
from probnum.problems.zoo.linalg import random_linear_system

import pytest_cases

cases_matrices = ".matrices"


@pytest_cases.parametrize_with_cases("matrix", cases=cases_matrices, scope="module")
def case_linsys(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
) -> problems.LinearSystem:
    """Linear system."""
    rng_state = backend.random.rng_state(abs(hash(matrix)))
    return random_linear_system(rng_state=rng_state, matrix=matrix)


@pytest_cases.parametrize_with_cases(
    "spd_matrix",
    cases=cases_matrices,
    has_tag=["symmetric", "positive_definite"],
    scope="module",
)
def case_spd_linsys(
    spd_matrix: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
) -> problems.LinearSystem:
    """Linear system with symmetric positive definite matrix."""
    rng_state = backend.random.rng_state(abs(hash(spd_matrix)))
    return random_linear_system(rng_state=rng_state, matrix=spd_matrix)
