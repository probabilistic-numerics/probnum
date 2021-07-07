"""Linear system test cases."""

import pathlib
from typing import Union

import numpy as np
import pytest_cases
import scipy.sparse

from probnum import linops, problems

cases_matrices = ".matrices"


@pytest_cases.parametrize_with_cases("matrix", cases=cases_matrices)
def case_linsys(
    matrix: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    rng: np.random.Generator,
) -> problems.LinearSystem:
    """Linear system."""
    if hasattr(matrix, "valuegetter"):
        matrix = matrix.valuegetter()

    return problems.LinearSystem.from_matrix(A=matrix, rng=rng)


@pytest_cases.parametrize_with_cases(
    "spd_matrix", cases=cases_matrices, has_tag=["symmetric", "positive_definite"]
)
def case_spd_linsys(
    spd_matrix: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    rng: np.random.Generator,
) -> problems.LinearSystem:
    """Linear system with symmetric positive definite matrix."""
    if hasattr(spd_matrix, "valuegetter"):
        spd_matrix = spd_matrix.valuegetter()

    return problems.LinearSystem.from_matrix(A=spd_matrix, rng=rng)
