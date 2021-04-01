import pathlib
from typing import Union

import numpy as np
import pytest
import pytest_cases

import probnum as pn

case_modules = [
    ".test_linops_cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "test_linops_cases").glob("*_cases.py")
]


@pytest.fixture(
    scope="function",
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def random_state(request) -> np.random.RandomState:
    return np.random.RandomState(seed=request.param)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_case_valid(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    assert isinstance(linop, pn.linops.LinearOperator)
    assert isinstance(matrix, np.ndarray)

    assert linop.ndim == matrix.ndim == 2
    assert linop.shape == matrix.shape
    assert linop.dtype == matrix.dtype


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matvec(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    random_state: np.random.RandomState,
):
    vec = random_state.normal(size=linop.shape[1])

    linop_matvec = linop @ vec
    matrix_matvec = matrix @ vec

    assert linop_matvec.ndim == 1
    assert linop_matvec.shape == matrix_matvec.shape
    assert linop_matvec.dtype == matrix_matvec.dtype

    np.testing.assert_allclose(linop_matvec, matrix_matvec)


@pytest.mark.parametrize("n", [1, 2, 15, 100])
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matmat(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    random_state: np.random.RandomState,
    n: int,
):
    mat = random_state.normal(size=(linop.shape[1], n))

    linop_matmat = linop @ mat
    matrix_matmat = matrix @ mat

    assert linop_matmat.ndim == 2
    assert linop_matmat.shape == matrix_matmat.shape
    assert linop_matmat.dtype == matrix_matmat.dtype

    np.testing.assert_allclose(linop_matmat, matrix_matmat)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matmat_shape_mismatch(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
):
    mat = np.zeros((linop.shape[1] + 1, 10))

    with pytest.raises(Exception) as excinfo:
        matrix @ mat  # pylint: disable=pointless-statement

    with pytest.raises(excinfo.type):
        linop @ mat  # pylint: disable=pointless-statement

    mat = np.zeros((1, linop.shape[1], 10))

    with pytest.raises(ValueError):
        linop @ mat  # pylint: disable=pointless-statement


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatvec(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    random_state: np.random.RandomState,
):
    vec = random_state.normal(size=linop.shape[0])

    linop_matvec = np.conj(vec) @ linop
    matrix_matvec = np.conj(vec) @ matrix

    assert linop_matvec.ndim == 1
    assert linop_matvec.shape == matrix_matvec.shape
    assert linop_matvec.dtype == matrix_matvec.dtype

    np.testing.assert_allclose(linop_matvec, matrix_matvec)


@pytest.mark.parametrize("n", [1, 2, 15, 100])
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatmat(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    random_state: np.random.RandomState,
    n: int,
):
    mat = random_state.normal(size=(n, linop.shape[0]))

    linop_matmat = np.conj(mat) @ linop
    matrix_matmat = np.conj(mat) @ matrix

    assert linop_matmat.ndim == 2
    assert linop_matmat.shape == matrix_matmat.shape
    assert linop_matmat.dtype == matrix_matmat.dtype

    np.testing.assert_allclose(linop_matmat, matrix_matmat)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatmat_shape_mismatch(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
):
    mat = np.zeros((10, linop.shape[0] + 1))

    with pytest.raises(Exception) as excinfo:
        mat @ matrix  # pylint: disable=pointless-statement

    with pytest.raises(excinfo.type):
        mat @ linop  # pylint: disable=pointless-statement

    mat = np.zeros((1, 10, linop.shape[0]))

    with pytest.raises(ValueError):
        mat[None, :, :] @ linop  # pylint: disable=pointless-statement


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_todense(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    linop_dense = linop.todense()

    assert isinstance(linop_dense, np.ndarray)
    assert linop_dense.shape == matrix.shape
    assert linop_dense.dtype == matrix.dtype

    np.testing.assert_allclose(linop_dense, matrix)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rank(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    linop_rank = linop.rank()
    matrix_rank = np.linalg.matrix_rank(matrix)

    assert isinstance(linop_rank, np.intp)
    assert linop_rank.shape == ()
    assert linop_rank.dtype == matrix_rank.dtype

    assert linop_rank == matrix_rank


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_eigvals(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        linop_eigvals = linop.eigvals()
        matrix_eigvals = np.linalg.eigvals(matrix)

        assert isinstance(linop_eigvals, np.ndarray)
        assert linop_eigvals.shape == matrix_eigvals.shape
        assert linop_eigvals.dtype == matrix_eigvals.dtype

        np.testing.assert_allclose(linop_eigvals, matrix_eigvals)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.eigvals()


@pytest.mark.parametrize("p", [None, 2])
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_cond(linop: pn.linops.LinearOperator, matrix: np.ndarray, p: Union[None, int]):
    linop_cond = linop.cond(p=p)
    matrix_cond = np.linalg.cond(matrix, p=p)

    assert isinstance(linop_cond, np.number)
    assert linop_cond.shape == ()
    assert linop_cond.dtype == matrix_cond.dtype

    np.testing.assert_allclose(linop_cond, matrix_cond)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_det(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        linop_det = linop.det()
        matrix_det = np.linalg.det(matrix)

        assert isinstance(linop_det, np.number)
        assert linop_det.shape == ()
        assert linop_det.dtype == matrix_det.dtype

        np.testing.assert_allclose(linop_det, matrix_det)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.det()


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_logabsdet(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        linop_logabsdet = linop.logabsdet()
        _, matrix_logabsdet = np.linalg.slogdet(matrix)

        assert isinstance(linop_logabsdet, np.inexact)
        assert linop_logabsdet.shape == ()
        assert linop_logabsdet.dtype == matrix_logabsdet.dtype

        np.testing.assert_allclose(linop_logabsdet, matrix_logabsdet)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.logabsdet()


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_trace(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        linop_trace = linop.trace()
        matrix_trace = np.trace(matrix)

        assert isinstance(linop_trace, np.number)
        assert linop_trace.shape == ()
        assert linop_trace.dtype == matrix_trace.dtype

        np.testing.assert_allclose(linop_trace, matrix_trace)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.trace()


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_transpose(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    linop_transpose = linop.T
    matrix_transpose = matrix.T

    assert isinstance(linop_transpose, pn.linops.LinearOperator)
    assert linop_transpose.shape == matrix_transpose.shape
    assert linop_transpose.dtype == matrix_transpose.dtype

    np.testing.assert_allclose(linop_transpose.todense(), matrix_transpose)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_inv(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        try:
            linop_inv = linop.inv()
        except np.linalg.LinAlgError:
            return

        matrix_inv = np.linalg.inv(matrix)

        assert isinstance(linop_inv, pn.linops.LinearOperator)
        assert linop_inv.shape == matrix_inv.shape
        assert linop_inv.dtype == matrix_inv.dtype

        np.testing.assert_allclose(linop_inv.todense(), matrix_inv, atol=1e-12)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.inv()
