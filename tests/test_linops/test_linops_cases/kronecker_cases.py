import itertools
from typing import Tuple, Union

import numpy as np
import pytest
import pytest_cases

import probnum as pn
from probnum.problems.zoo.linalg import random_spd_matrix

spd_matrices = (
    pn.linops.Identity(shape=(1, 1)),
    np.array([[1.0, -2.0], [-2.0, 5.0]]),
    random_spd_matrix(np.random.default_rng(597), dim=10),
    random_spd_matrix(np.random.default_rng(597), dim=13),
)


@pytest.mark.parametrize(
    "A,B",
    [
        (np.array([[2, -3.5], [12, 6.5]]), np.eye(3)),
        (np.array([[1, -2], [-2.2, 5]]), np.array([[1, -3], [0, -0.5]])),
        (np.array([[4, 1, 4], [2, 3, 2]]), np.array([[-1, 4], [2, 1]])),
        (np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9]]), np.array([[1, 4]])),
        (np.array([[4, 1], [2, 3]]), np.array([[-1, 4], [2, 1]])),
        (
            np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9], [1, 0, 2]]),
            np.array([[1, 4, 0], [-3, -0.4, -100], [0.18, -2, 10]]),
        ),
        (
            np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9], [1, 0, 2]]),
            np.array([[1, 4, 0], [-3, -0.4, -100]]),
        ),
    ],
)
def case_kronecker(
    A: np.ndarray, B: np.ndarray
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.Kronecker(A, B)
    matrix = np.kron(A, B)

    return linop, matrix


@pytest_cases.case(tags=["square", "symmetric", "positive-definite"])
@pytest_cases.parametrize("A,B", itertools.product(spd_matrices, spd_matrices))
def case_kronecker_positive_definite(
    A: Union[np.ndarray, pn.linops.LinearOperator],
    B: Union[np.ndarray, pn.linops.LinearOperator],
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    A = pn.linops.aslinop(A)
    A.is_symmetric = True

    B = pn.linops.aslinop(B)
    B.is_symmetric = True

    linop = pn.linops.Kronecker(A, B)
    matrix = np.kron(A.todense(), B.todense())

    return linop, matrix


@pytest.mark.parametrize(
    "A,B",
    [
        (np.array([[1, -2], [-2.2, 5]]), np.array([[1, -3], [0, -0.5]])),
        (np.array([[4, 1], [2, 3]]), np.array([[-1, 4], [2, 1]])),
        (
            np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9], [1, 0, 2]]),
            np.array([[1, 4, 0], [-3, -0.4, -100], [0.18, -2, 10]]),
        ),
    ],
)
@pytest_cases.case(tags=["symmetric_kronecker"])
def case_symmetric_kronecker(
    A: np.ndarray, B: np.ndarray
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.SymmetricKronecker(A, B)
    matrix = (np.kron(A, B) + np.kron(B, A)) / 2

    return linop, matrix


@pytest.mark.parametrize(
    "A",
    [
        np.array([[1, -2], [-2.2, 5]]),
        np.array([[1, -3], [0, -0.5]]),
        np.array([[4, 1], [2, 3]]),
        np.array([[-1, 4], [2, 1]]),
        np.array([[0.4, 2, 0.8], [-0.4, 0, -0.9], [1, 0, 2]]),
        np.array([[1, 4, 0], [-3, -0.4, -100], [0.18, -2, 10]]),
    ],
)
@pytest_cases.case(tags=["symmetric_kronecker"])
def case_symmetric_kronecker_identical_factors(
    A: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.SymmetricKronecker(A)
    matrix = np.kron(A, A)

    return linop, matrix


@pytest_cases.case(
    tags=[
        "symmetric_kronecker",
        "square",
        "symmetric",
        "positive-definite",
    ]
)
@pytest_cases.parametrize("A", spd_matrices)
def case_symmetric_kronecker_identical_factors_positive_definite(
    A: Union[np.ndarray, pn.linops.LinearOperator],
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    A = pn.linops.aslinop(A)
    A.is_symmetric = True

    linop = pn.linops.SymmetricKronecker(A)
    matrix = np.kron(A.todense(), A.todense())

    return linop, matrix
