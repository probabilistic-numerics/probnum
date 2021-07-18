"""Linear system test cases."""

from typing import Union

import numpy as np
import pytest_cases
import scipy.sparse

from probnum import linops, problems
from probnum.problems.zoo.linalg import random_linear_system

cases_matrices = ".matrices"


@pytest_cases.parametrize_with_cases("matrix", cases=cases_matrices)
def case_linsys(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    rng: np.random.Generator,
) -> problems.LinearSystem:
    """Linear system."""
    return random_linear_system(rng=rng, matrix=matrix)


@pytest_cases.parametrize_with_cases(
    "spd_matrix", cases=cases_matrices, has_tag=["symmetric", "positive_definite"]
)
def case_spd_linsys(
    spd_matrix: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    rng: np.random.Generator,
) -> problems.LinearSystem:
    """Linear system with symmetric positive definite matrix."""
    return random_linear_system(rng=rng, matrix=spd_matrix)
