from typing import Tuple

import numpy as np
import pytest_cases

import probnum as pn
from probnum.linops._arithmetic_fallbacks import ScaledLinearOperator, SumLinearOperator
from probnum.problems.zoo.linalg import random_spd_matrix

spd_matrix_pairs = [
    (
        random_spd_matrix(np.random.default_rng(n + 9872), dim=n),
        random_spd_matrix(np.random.default_rng(n + 1231), dim=n),
    )
    for n in [1, 2, 3, 5, 8]
]


@pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
@pytest_cases.parametrize("A", [A for A, _ in spd_matrix_pairs])
@pytest_cases.parametrize("scalar", (4.2,))
def case_scaled_linop_positive_definite(A: np.ndarray, scalar: float):
    matrix = scalar * A

    A = pn.linops.aslinop(A)
    A.is_symmetric = True

    return ScaledLinearOperator(A, scalar), matrix


@pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
@pytest_cases.parametrize("A,B", spd_matrix_pairs)
def case_sum_linop_positive_definite(
    A: np.ndarray, B: np.ndarray
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    matrix = A + B

    A = pn.linops.aslinop(A)
    A.is_symmetric = True

    B = pn.linops.aslinop(B)
    B.is_symmetric = True

    return SumLinearOperator(A, B), matrix
