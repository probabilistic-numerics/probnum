from typing import Tuple

import numpy as np
import pytest

import probnum as pn

matrices = [
    np.array([[-1.5, 3], [0, -230]]),
    np.array([[2, 0], [1, 3]]),
    np.array([[2, 0, -1.5], [1, 3, -230]]),
]


@pytest.mark.parametrize("matrix", matrices)
def case_matvec(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.LinearOperator(
        shape=matrix.shape, dtype=matrix.dtype, matvec=lambda x: matrix @ x
    )

    return linop, matrix


@pytest.mark.parametrize("matrix", matrices)
def case_matrix(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.MatrixMult(matrix), matrix


@pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
def case_identity(n: int) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Identity(shape=n), np.eye(n)
