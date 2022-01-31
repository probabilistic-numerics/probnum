"""Functions defining useful inner products."""

from typing import Optional, Union

import numpy as np

from probnum import linops


def inner_product(
    v: np.ndarray,
    w: np.ndarray,
    A: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
) -> np.ndarray:
    r"""Inner product :math:`\langle v, w \rangle_A := v^T A w`.

    For arrays the function computes the inner product over the last axes of the
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
        *shape=(\*v.shape[:-1], \*w.shape[:-1])* -- Inner product of ``v`` and ``w``. If they are both 1-D arrays then a scalar is returned; otherwise an array is returned.
    """
    w = w[..., :, None]

    if A is None:
        vw_inprod = np.dot(v, w)
    else:
        vw_inprod = np.dot(v, A @ w)

    return np.squeeze(vw_inprod, axis=(-1))


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
        Symmetric positive (semi-)definite matrix defining the geometry.
    axis
        Specifies the axis along which to compute the vector norms.

    Returns
    -------
    norm :
        Vector norm of ``v`` along the given.
    """

    if A is None:
        return np.linalg.norm(v, ord=2, axis=axis, keepdims=False)

    v_moved_axis = np.moveaxis(v, axis, -1)
    return np.sqrt(inner_product(v_moved_axis, v_moved_axis, A))
