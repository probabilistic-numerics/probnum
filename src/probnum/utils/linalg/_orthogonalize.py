"""Orthogonalization of vectors."""

from typing import Callable, Iterable, Optional, Union

import numpy as np

from probnum import linops

from ._inner_product import induced_norm
from ._inner_product import inner_product as inner_product_fn


def gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    inner_product: Optional[
        Union[
            np.ndarray,
            linops.LinearOperator,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ]
    ] = None,
    normalize: bool = False,
) -> np.ndarray:
    r"""Orthogonalize a vector with respect to an orthogonal basis and inner product.

    Computes a vector :math:`v'` such that :math:`\langle v', b_i \rangle = 0` for
    all basis vectors :math:`b_i \in B` in the orthogonal basis.

    Parameters
    ----------
    v
        Vector to orthogonalize.
    orthogonal_basis
        Orthogonal basis.
    inner_product
        Inner product defining orthogonality. Can be either a :class`numpy.ndarray` or a :class:`Callable`
        defining the inner product. Defaults to the euclidean inner product.
    normalize
        Normalize the output vector, s.t. :math:`\langle v', v' \rangle = 1`.

    Returns
    -------
    v_orth :
        Orthogonalized vector.
    """
    if inner_product is None:
        inprod_fn = inner_product_fn
        norm_fn = induced_norm
    elif isinstance(inner_product, (np.ndarray, linops.LinearOperator)):
        inprod_fn = lambda v, w: inner_product_fn(v, w, A=inner_product)
        norm_fn = lambda v: induced_norm(v, A=inner_product)
    else:
        inprod_fn = inner_product
        norm_fn = lambda v: np.sqrt(inprod_fn(v, v))

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod_fn(u, v) / inprod_fn(u, u)) * u

    if normalize:
        v_orth /= norm_fn(v_orth)

    return v_orth


def modified_gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    inner_product: Optional[
        Union[
            np.ndarray,
            linops.LinearOperator,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ]
    ] = None,
    normalize: bool = False,
) -> np.ndarray:
    r"""Stabilized Gram-Schmidt process.

    Computes a vector :math:`v'` such that :math:`\langle v', b_i \rangle = 0` for
    all basis vectors :math:`b_i \in B` in the orthogonal basis in a numerically stable fashion.

    Parameters
    ----------
    v
        Vector to orthogonalize.
    orthogonal_basis
        Orthogonal basis.
    inner_product
        Inner product defining orthogonality. Can be either a :class`numpy.ndarray` or a :class:`Callable`
        defining the inner product. Defaults to the euclidean inner product.
    normalize
        Normalize the output vector, s.t. :math:`\langle v', v' \rangle = 1`.

    Returns
    -------
    v_orth :
        Orthogonalized vector.
    """
    if inner_product is None:
        inprod_fn = inner_product_fn
        norm_fn = induced_norm
    elif isinstance(inner_product, (np.ndarray, linops.LinearOperator)):
        inprod_fn = lambda v, w: inner_product_fn(v, w, A=inner_product)
        norm_fn = lambda v: induced_norm(v, A=inner_product)
    else:
        inprod_fn = inner_product
        norm_fn = lambda v: np.sqrt(inprod_fn(v, v))

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod_fn(u, v_orth) / inprod_fn(u, u)) * u

    if normalize:
        v_orth /= norm_fn(v_orth)

    return v_orth


def double_gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    inner_product: Optional[
        Union[
            np.ndarray,
            linops.LinearOperator,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ]
    ] = None,
    normalize: bool = False,
) -> np.ndarray:
    r"""Perform the modified Gram-Schmidt process twice.

    Computes a vector :math:`v'` such that :math:`\langle v', b_i \rangle = 0` for
    all basis vectors :math:`b_i \in B` in the orthogonal basis. This performs the
    modified Gram-Schmidt orthogonalization process twice, which is generally more
    stable than just reorthogonalizing once. [1]_ [2]_

    Parameters
    ----------
    v
        Vector to orthogonalize.
    orthogonal_basis
        Orthogonal basis.
    inner_product
        Inner product defining orthogonality. Can be either a :class`numpy.ndarray` or a :class:`Callable`
        defining the inner product. Defaults to the euclidean inner product.
    normalize
        Normalize the output vector, s.t. :math:`\langle v', v' \rangle = 1`.

    Returns
    -------
    v_orth :
        Orthogonalized vector.

    References
    ----------
    .. [1] L. Giraud, J. Langou, M. Rozloznik, and J. van den Eshof, Rounding error
           analysis of the classical Gram-Schmidt orthogonalization process, Numer. Math., 101 (2005), pp. 87–100
    .. [2] L. Giraud, J. Langou, and M. Rozloznik, The loss of orthogonality in the
           Gram-Schmidt orthogonalization process, Comput. Math. Appl., 50 (2005)
    """
    v_orth = modified_gram_schmidt(
        v=v,
        orthogonal_basis=orthogonal_basis,
        inner_product=inner_product,
        normalize=normalize,
    )
    return modified_gram_schmidt(
        v=v_orth,
        orthogonal_basis=orthogonal_basis,
        inner_product=inner_product,
        normalize=normalize,
    )
