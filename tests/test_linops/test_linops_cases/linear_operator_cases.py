from typing import Tuple

import numpy as np
import pytest
import pytest_cases
import scipy.sparse

import probnum as pn
from probnum.problems.zoo.linalg import random_spd_matrix

matrices = [
    np.array([[-1.5, 3], [0, -230]]),
    np.array([[2, 0], [1, 3]]),
    np.array([[2, 0, -1.5], [1, 3, -230]]),
]
spd_matrices = [
    np.array([[1.0]]),
    np.array([[1.0, -2.0], [-2.0, 5.0]]),
    random_spd_matrix(np.random.default_rng(597), dim=10),
]
lower_triangular_matrices = [
    np.array([[5.0]]),
    np.array([[1.0, 0.0], [2.0, 3.0]]),
    np.array([[4.0, 0.0, 0.0], [48.0, 60.0, 0.0], [21.0, 39.0, 7.0]]),
]
upper_triangular_matrices = [mat.T for mat in lower_triangular_matrices]


@pytest.mark.parametrize("matrix", matrices)
def case_matvec(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    @pn.linops.LinearOperator.broadcast_matvec
    def _matmul(vec: np.ndarray):
        return matrix @ vec

    linop = pn.linops.LambdaLinearOperator(
        shape=matrix.shape, dtype=matrix.dtype, matmul=_matmul
    )

    return linop, matrix


@pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
@pytest_cases.parametrize("matrix", spd_matrices)
def case_matvec_spd(matrix: np.ndarray):
    @pn.linops.LinearOperator.broadcast_matvec
    def _matmul(vec: np.ndarray):
        return matrix @ vec

    linop = pn.linops.LambdaLinearOperator(
        shape=matrix.shape, dtype=matrix.dtype, matmul=_matmul
    )

    linop.is_symmetric = True

    return linop, matrix


@pytest.mark.parametrize("matrix", matrices)
def case_matrix(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Matrix(matrix), matrix


@pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
@pytest_cases.parametrize("matrix", spd_matrices)
def case_matrix_spd(matrix: np.ndarray) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.Matrix(matrix)
    linop.is_symmetric = True

    return linop, matrix


@pytest_cases.case(tags=("square", "lower-triangular"))
@pytest_cases.parametrize("matrix", lower_triangular_matrices)
def case_matrix_lower_triangular(
    matrix: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.Matrix(matrix)
    linop.is_lower_triangular = True

    return linop, matrix


@pytest_cases.case(tags=("square", "upper-triangular"))
@pytest_cases.parametrize("matrix", upper_triangular_matrices)
def case_matrix_upper_triangular(
    matrix: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.Matrix(matrix)
    linop.is_upper_triangular = True

    return linop, matrix


@pytest_cases.case(tags=("square", "symmetric"))
def case_matrix_symmetric_indefinite() -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    matrix = np.diag((2.1, 1.3, -0.5))

    linop = pn.linops.aslinop(matrix)
    linop.is_symmetric = True

    return linop, matrix


@pytest_cases.case(tags=("square", "symmetric", "positive-definite"))
@pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
def case_identity(n: int) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Identity(shape=n), np.eye(n)


@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (6, 5)])
def case_zero(shape: tuple) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Zero(shape=shape), np.zeros(shape)


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


@pytest.mark.parametrize("rng", [np.random.default_rng(422)])
def case_inverse(
    rng: np.random.Generator,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    N = 21

    v = rng.uniform(0.2, 0.5, N)

    linop = pn.linops.LambdaLinearOperator(
        shape=(N, N),
        dtype=np.double,
        matmul=lambda x: 2.0 * x + v[:, None] @ (v[None, :] @ x),
    )

    return linop.inv(), linop.inv().todense()
