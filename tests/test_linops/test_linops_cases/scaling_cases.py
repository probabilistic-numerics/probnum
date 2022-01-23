from typing import Tuple

import numpy as np
import pytest_cases

import probnum as pn


@pytest_cases.case(tags=["square", "symmetric", "indefinite"])
@pytest_cases.parametrize(
    "diagonal",
    [
        np.array([1.0, 2, -3]),
    ],
)
def case_anisotropic_scaling(
    diagonal: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(diagonal), np.diag(diagonal)


@pytest_cases.case(tags=["square", "symmetric", "positive-definite"])
@pytest_cases.parametrize(
    "diagonal",
    [
        np.array([2.0, 5.0, 8.0]),
    ],
)
def case_positive_anisotropic_scaling(
    diagonal: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(diagonal), np.diag(diagonal)


@pytest_cases.case(tags=["square", "symmetric", "singular", "indefinite"])
@pytest_cases.parametrize(
    "diagonal",
    [
        np.array([1.0, 2.0, 0.0, -1.0]),
    ],
)
def case_singular_anisotropic_scaling(
    diagonal: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(diagonal), np.diag(diagonal)


@pytest_cases.case(tags=["square", "symmetric", "positive-definite"])
@pytest_cases.parametrize("n", [3, 4, 8, 12, 15])
@pytest_cases.parametrize(
    "scalar",
    [
        1.0,  # Identity
        2.8,  # double dtype
        5,  # Integer dtype
    ],
)
def case_positive_isotropic_scaling(
    n: int, scalar
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(scalar, shape=n), np.diag(np.full((n,), scalar))


@pytest_cases.case(
    tags=[
        "square",
        "symmetric",
        "singular",
        "positive-semidefinite",
        "negative-semidefinite",
    ]
)
@pytest_cases.parametrize("n", [3, 4, 8, 12, 15])
def case_singular_isotropic_scaling(n: int):
    return pn.linops.Scaling(0.0, shape=n), np.zeros((n,), dtype=np.double)


@pytest_cases.case(tags=["square", "symmetric", "negative-definite"])
@pytest_cases.parametrize("n", [3, 4, 8, 12, 15])
@pytest_cases.parametrize(
    "scalar",
    [
        -8.3,
    ],
)
def case_negative_isotropic_scaling(
    n: int, scalar
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(scalar, shape=n), np.diag(np.full((n,), scalar))
