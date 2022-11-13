"""Implementation of linear algebra functionality in NumPy."""

import functools
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from numpy.linalg import eigh, eigvalsh, solve, svd
import scipy.linalg


def vector_norm(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal["inf", "-inf"]] = 2,
) -> np.ndarray:
    return np.asarray(np.linalg.norm(x=x, ord=ord, keepdims=keepdims, axis=axis))


def matrix_norm(x: np.ndarray, /, *, keepdims: bool = False, ord="fro") -> np.ndarray:
    return np.asarray(np.linalg.norm(x=x, ord=ord, keepdims=keepdims, axis=(-2, -1)))


def cholesky(x: np.ndarray, /, *, upper: bool = False) -> np.ndarray:
    try:
        L = np.linalg.cholesky(x)

        return np.conj(L.swapaxes(-2, -1)) if upper else L
    except np.linalg.LinAlgError:
        return (np.triu if upper else np.tril)(np.full_like(x, np.nan))


def solve_triangular(
    A: np.ndarray,
    b: np.ndarray,
    *,
    transpose: bool = False,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> np.ndarray:
    if b.ndim in (1, 2):
        return scipy.linalg.solve_triangular(
            A,
            b,
            trans=1 if transpose else 0,
            lower=lower,
            unit_diagonal=unit_diagonal,
        )

    return _matmul_broadcasting(
        functools.partial(
            scipy.linalg.solve_triangular,
            A,
            trans=1 if transpose else 0,
            lower=lower,
            unit_diagonal=unit_diagonal,
        ),
        b,
    )


def solve_cholesky(
    cholesky: np.ndarray,
    b: np.ndarray,
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
):
    if b.ndim in (1, 2):
        return scipy.linalg.cho_solve(
            (cholesky, lower),
            b,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )

    return _matmul_broadcasting(
        functools.partial(
            scipy.linalg.cho_solve,
            (cholesky, lower),
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        ),
        b,
    )


def _matmul_broadcasting(
    matmul_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    # In order to apply __matmul__ broadcasting, we need to reshape the stack of
    # matrices `x` into a matrix whose first axis corresponds to the penultimate axis in
    # the matrix stack and whose second axis is a flattened/raveled representation of
    # all the remaining axes

    # We can handle a stack of vectors in a simplified manner
    stack_of_vectors = x.shape[-1] == 1

    if stack_of_vectors:
        x_batch_first = x[..., 0]
    else:
        x_batch_first = np.swapaxes(x, -2, -1)

    x_batch_last = np.array(x_batch_first.T, copy=False, order="F")

    # Flatten the trailing axes and remember shape to undo flattening operation later
    unflatten_shape = x_batch_last.shape[1:]
    x_flat_batch_last = x_batch_last.reshape(
        (x_batch_last.shape[0], -1),
        order="F",
    )

    assert x_flat_batch_last.flags.f_contiguous

    res_flat_batch_last = np.array(
        matmul_fn(x_flat_batch_last),
        copy=False,
        order="F",
    )

    # Undo flattening operation
    res_batch_last = res_flat_batch_last.reshape((-1,) + unflatten_shape, order="F")

    res_batch_first = res_batch_last.T

    if stack_of_vectors:
        return res_batch_first[..., None]

    return np.swapaxes(res_batch_first, -2, -1)


def qr(
    x: np.ndarray, /, *, mode: Literal["reduced", "complete", "r"] = "reduced"
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "r":
        r = np.linalg.qr(x, mode=mode)
        q = None
    else:
        q, r = np.linalg.qr(x, mode=mode)

    return q, r
