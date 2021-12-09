"""Functions defining useful inner products."""

from typing import Optional, Union

import numpy as np

from probnum import linops


def euclidean_inprod(
    v: np.ndarray,
    w: np.ndarray,
    A: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
) -> np.ndarray:
    r"""(Modified) Euclidean inner product :math:`\langle v, w \rangle_A := v^T A w`.

    Parameters
    ----------
    v
        First vector.
    w
        Second vector.
    A
        Symmetric positive (semi-)definite matrix defining the geometry.

    Returns
    -------
    inprod
        Inner product.
    """

    v_T = v[..., None, :]
    w = w[..., :, None]

    if A is None:
        vw_inprod = v_T @ w
    else:
        vw_inprod = v_T @ (A @ w)

    return np.squeeze(vw_inprod, axis=(-2, -1))


def euclidean_norm(
    v: np.ndarray,
    A: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
) -> np.ndarray:
    r"""(Modified) Euclidean norm :math:`\lVert v \rVert_A := \sqrt{v^T A v}`.

    Parameters
    ----------
    v
        Vector.
    A
        Symmetric positive (semi-)definite matrix defining the geometry.

    Returns
    -------
    norm
        Vector norm.
    """

    if A is None:
        return np.linalg.norm(v, ord=2, axis=-1, keepdims=False)

    return np.sqrt(euclidean_inprod(v, v, A))
