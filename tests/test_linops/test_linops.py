import pathlib
from typing import Optional, Tuple, Union

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
def rng(request) -> np.random.Generator:
    return np.random.default_rng(seed=request.param)


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
    rng: np.random.Generator,
):
    vec = rng.normal(size=linop.shape[1])

    linop_matvec = linop @ vec
    matrix_matvec = matrix @ vec

    assert linop_matvec.ndim == 1
    assert linop_matvec.shape == matrix_matvec.shape
    assert linop_matvec.dtype == matrix_matvec.dtype

    np.testing.assert_allclose(linop_matvec, matrix_matvec)


@pytest.mark.parametrize("ncols", [1, 2, 15, 100])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_matmat(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    rng: np.random.Generator,
    ncols: int,
    order: str,
):
    mat = np.asarray(rng.normal(size=(linop.shape[1], ncols)), order=order)

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


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatvec(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    rng: np.random.Generator,
):
    vec = rng.normal(size=linop.shape[0])

    linop_matvec = vec @ linop
    matrix_matvec = vec @ matrix

    assert linop_matvec.ndim == 1
    assert linop_matvec.shape == matrix_matvec.shape
    assert linop_matvec.dtype == matrix_matvec.dtype

    np.testing.assert_allclose(linop_matvec, matrix_matvec)


@pytest.mark.parametrize("nrows", [1, 2, 15, 100])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_rmatmat(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    rng: np.random.Generator,
    nrows: int,
    order: str,
):
    mat = np.asarray(rng.normal(size=(nrows, linop.shape[0])), order=order)

    linop_matmat = mat @ linop
    matrix_matmat = mat @ matrix

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


@pytest.mark.parametrize(
    "shape",
    [
        # axis=-2 (__matmul__)
        (1, 1, None, 1),
        (2, 8, None, 2),
        # axis=-2 (__matmul__ on arr[..., np.newaxis])
        (1, 1, 1, None),
        (5, 2, 3, None),
        (3, 5, 3, None),
        # axis < -2
        (1, None, 1, 1),
        (5, None, 3, 3),
        (None, 3, 4, 2),
    ],
)
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_call(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    rng: np.random.Generator,
    shape: Tuple[Optional[int], ...],
):
    axis = shape.index(None) - len(shape)
    shape = tuple(entry if entry is not None else linop.shape[1] for entry in shape)

    arr = rng.normal(size=shape)

    linop_call = linop(arr, axis=axis)
    matrix_call = np.moveaxis(np.tensordot(matrix, arr, axes=(-1, axis)), 0, axis)

    assert linop_call.ndim == 4
    assert linop_call.shape == matrix_call.shape
    assert linop_call.dtype == matrix_call.dtype

    np.testing.assert_allclose(linop_call, matrix_call)


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


@pytest.mark.parametrize("p", [None, 1, 2, np.inf, "fro", -1, -2, -np.inf])
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_cond(
    linop: pn.linops.LinearOperator, matrix: np.ndarray, p: Union[None, int, float, str]
):
    if linop.is_square:
        linop_cond = linop.cond(p=p)
        matrix_cond = np.linalg.cond(matrix, p=p)

        assert isinstance(linop_cond, np.inexact)
        assert linop_cond.shape == ()
        assert linop_cond.dtype == matrix_cond.dtype

        try:
            np.testing.assert_allclose(linop_cond, matrix_cond)
        except AssertionError as e:
            if p == -2 and 0 < linop.rank() < linop.shape[0] and linop_cond == np.inf:
                # `np.linalg.cond` returns 0.0 for p = -2 if the matrix is singular but
                # not zero. This is a bug.
                pass
            else:
                raise e
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.cond(p=p)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_det(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        linop_det = linop.det()
        matrix_det = np.linalg.det(matrix)

        assert isinstance(linop_det, np.inexact)
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
def test_conjugate(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    linop_conj = linop.conj()
    matrix_conj = matrix.conj()

    assert isinstance(linop_conj, pn.linops.LinearOperator)
    assert linop_conj.shape == matrix_conj.shape
    assert linop_conj.dtype == matrix_conj.dtype

    np.testing.assert_allclose(linop_conj.todense(), matrix_conj)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_transpose(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    linop_transpose = linop.T
    matrix_transpose = matrix.T

    assert isinstance(linop_transpose, pn.linops.LinearOperator)
    assert linop_transpose.shape == matrix_transpose.shape
    assert linop_transpose.dtype == matrix_transpose.dtype

    np.testing.assert_allclose(linop_transpose.todense(), matrix_transpose)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_adjoint(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    linop_adjoint = linop.H
    matrix_adjoint = matrix.T.conj()

    assert isinstance(linop_adjoint, pn.linops.LinearOperator)
    assert linop_adjoint.shape == matrix_adjoint.shape
    assert linop_adjoint.dtype == matrix_adjoint.dtype

    np.testing.assert_allclose(linop_adjoint.todense(), matrix_adjoint)


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_inv(linop: pn.linops.LinearOperator, matrix: np.ndarray):
    if linop.is_square:
        expected_exception = None

        try:
            matrix_inv = np.linalg.inv(matrix)
        except Exception as e:  # pylint: disable=broad-except
            expected_exception = e

        if expected_exception is None:
            linop_inv = linop.inv()

            assert isinstance(linop_inv, pn.linops.LinearOperator)
            assert linop_inv.shape == matrix_inv.shape
            assert linop_inv.dtype == matrix_inv.dtype

            np.testing.assert_allclose(linop_inv.todense(), matrix_inv, atol=1e-12)
        else:
            with pytest.raises(type(expected_exception)):
                linop.inv()
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.inv()
