from typing import Tuple

import numpy as np
import pytest_cases

import probnum as pn
from probnum.linops._arithmetic_fallbacks import (
    NegatedLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
)
from probnum.problems.zoo.linalg import random_spd_matrix

square_matrix_pairs = [
    (
        np.random.default_rng(n + 478).standard_normal((n, n)),
        np.random.default_rng(n + 267).standard_normal((n, n)),
    )
    for n in [1, 2, 3, 5, 8]
]

spd_matrix_pairs = [
    (
        random_spd_matrix(np.random.default_rng(n + 9872), dim=n),
        random_spd_matrix(np.random.default_rng(n + 1231), dim=n),
    )
    for n in [1, 2, 3, 5, 8]
]


@pytest_cases.case(tags=("square",))
@pytest_cases.parametrize("A", [A for A, _ in square_matrix_pairs])
@pytest_cases.parametrize("scalar", (4.2,))
def case_scaled_linop_square(A: np.ndarray, scalar: float):
    return ScaledLinearOperator(pn.linops.aslinop(A), scalar), scalar * A


@pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
@pytest_cases.parametrize("A", [A for A, _ in spd_matrix_pairs])
@pytest_cases.parametrize("scalar", (4.2,))
def case_scaled_linop_positive_definite(A: np.ndarray, scalar: float):
    matrix = scalar * A

    A = pn.linops.aslinop(A)
    A.is_symmetric = True

    return ScaledLinearOperator(A, scalar), matrix


@pytest_cases.case(tags=("square", "symmetric", "negative-definite"))
@pytest_cases.parametrize("A", [A for A, _ in spd_matrix_pairs])
def case_negated_linop_negative_definite(A: np.ndarray):
    matrix = -A

    A = pn.linops.aslinop(A)
    A.is_symmetric = True

    return NegatedLinearOperator(A), matrix


@pytest_cases.case(tags=("square"))
@pytest_cases.parametrize("A,B", spd_matrix_pairs)
def case_sum_linop_square(
    A: np.ndarray, B: np.ndarray
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return SumLinearOperator(pn.linops.aslinop(A), pn.linops.aslinop(B)), A + B


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
