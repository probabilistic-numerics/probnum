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
    r"""Inner product :math:`\langle v, w \rangle_A := v^T A w`.

    For n-d arrays the function computes the inner product over the last axis of the
    two arrays ``v`` and ``w``.

    Parameters
    ----------
    v
        First array.
    w
        Second array.
    A
        Symmetric positive (semi-)definite matrix defining the geometry.

    Returns
    -------
    inprod :
        Inner product(s) of ``v`` and ``w``.

    Notes
    -----
    Note that the broadcasting behavior of :func:`inner_product` differs from
    :func:`numpy.inner`. Rather it follows the broadcasting rules of
    :func:`numpy.matmul` in that n-d arrays are treated as stacks of vectors.
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
    r"""Induced norm :math:`\lVert v \rVert_A := \sqrt{v^T A v}`.

    Computes the induced norm over the given axis of the array.

    Parameters
    ----------
    v
        Array.
    A
        Symmetric positive (semi-)definite linear operator defining the geometry.
    axis
        Specifies the axis along which to compute the vector norms.

    Returns
    -------
    norm :
        Vector norm of ``v`` along the given ``axis``.
    """

    if A is None:
        return np.linalg.norm(v, ord=2, axis=axis, keepdims=False)

    v = np.moveaxis(v, axis, -1)
    w = np.squeeze(A @ v[..., :, None], axis=-1)

    return np.sqrt(np.sum(v * w, axis=-1))
