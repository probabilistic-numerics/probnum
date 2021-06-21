from typing import Tuple

import numpy as np
import pytest
import scipy.sparse

import probnum as pn

matrices = [
    np.array([[-1.5, 3], [0, -230]]),
    np.array([[2, 0], [1, 3]]),
    np.array([[2, 0, -1.5], [1, 3, -230]]),
]


@pytest.mark.parametrize("matrix", matrices)
def case_matvec(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    @pn.linops.LinearOperator.broadcast_matvec
    def _matmul(vec: np.ndarray):
        return matrix @ vec

    linop = pn.linops.LinearOperator(
        shape=matrix.shape, dtype=matrix.dtype, matmul=_matmul
    )

    return linop, matrix


@pytest.mark.parametrize("matrix", matrices)
def case_matrix(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Matrix(matrix), matrix


@pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
def case_identity(n: int) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Identity(shape=n), np.eye(n)


@pytest.mark.parametrize("rng", [np.random.default_rng(42)])
def case_sparse_matrix(
    rng: np.random.Generator,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    matrix = scipy.sparse.rand(
        10, 10, density=0.1, format="coo", dtype=np.double, random_state=rng
    )
    matrix.setdiag(2)
    matrix = matrix.tocsr()

    return pn.linops.Matrix(matrix), matrix.toarray()


@pytest.mark.parametrize("rng", [np.random.default_rng(42)])
def case_sparse_matrix_singular(
    rng: np.random.Generator,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    matrix = scipy.sparse.rand(
        10, 10, density=0.01, format="csr", dtype=np.double, random_state=rng
    )

    return pn.linops.Matrix(matrix), matrix.toarray()
