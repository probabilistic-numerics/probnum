"""Functions defining useful inner products."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from probnum import linops


def inner_product(
    v: np.ndarray,
    w: np.ndarray,
    A: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
) -> np.ndarray:
    r"""
    Computes the inner product over the last axis of the
    two arrays ``v`` and ``w``.

    Parameters
    ----------
    v
        First array, n-d array.
    w
        Second array, n-array.
    A
        Symmetric positive (semi-)definite matrix defining the geometry.

    Returns
    -------
    int or float depending on the dtype of v,w and A.

    Notes
    -----
    Note that the broadcasting behavior here follows that of :func:`numpy.matmul.
    """
    v_T = v[..., None, :]
    w = w[..., :, None]

    if A is None:
        vw_inprod = v_T @ w
    else:
        vw_inprod = v_T @ (A @ w)

    return np.squeeze(vw_inprod, axis=(-2, -1))


def induced_norm(
    v: np.ndarray,
    A: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
    axis: int = -1,
) -> np.ndarray:
    r"""
    Computes the induced norm over the given axis of the array.

    Parameters
    ----------
    v
     n-d array.
    A
        Symmetric positive (semi-)definite linear operator defining the geometry.
    axis
        Specifies the axis along which to compute the vector norms.

    Returns
    -------
    norm :
        Vector norm of ``v`` along the given ``axis``.
    """

    if A is not None:
        v = np.moveaxis(v, axis, -1)
        w = np.squeeze(A @ v[..., :, None], axis=-1)
        return np.sqrt(np.sum(v * w, axis=-1))
    else:
        return np.linalg.norm(v, ord=2, axis=axis, keepdims=False)


