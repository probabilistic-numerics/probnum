"""Linear system test cases."""

from typing import Union

import numpy as np
import pytest_cases

from probnum import linops, problems

from .matrices import *


@pytest_cases.parametrize_with_cases("spd_matrix", cases=SPDMatrix)
def case_spd_linsys(
    spd_matrix: Union[np.ndarray, linops.LinearOperator], rng: np.random.Generator
) -> problems.LinearSystem:
    """Linear system with symmetric positive definite matrix."""
    A = spd_matrix
    if hasattr(spd_matrix, "valuegetter"):
        A = spd_matrix.valuegetter()

    solution = rng.normal(size=A.shape[1])
    return problems.LinearSystem(A=A, b=A @ solution, solution=solution)
