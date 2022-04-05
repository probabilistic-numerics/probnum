"""Orthogonalization of vectors."""

from functools import partial
from typing import Callable, Iterable, Optional, Union

import numpy as np

from probnum import linops

from ._inner_product import induced_norm, inner_product as inner_product_fn


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
        Vector (or stack of vectors) to orthogonalize against ``orthogonal_basis``.
    orthogonal_basis
        Orthogonal basis.
    inner_product
        Inner product defining orthogonality. Can be either a :class`numpy.ndarray` or
        a :class:`Callable` defining the inner product. Defaults to the euclidean inner
        product.
    normalize
        Normalize the output vector, s.t. :math:`\langle v', v' \rangle = 1`.

    Returns
    -------
    v_orth :
        Orthogonalized vector.
    """
    orthogonal_basis = np.atleast_2d(orthogonal_basis)

    if inner_product is None:
        inprod_fn = inner_product_fn
        norm_fn = partial(induced_norm, axis=-1)
    elif isinstance(inner_product, (np.ndarray, linops.LinearOperator)):
        inprod_fn = lambda v, w: inner_product_fn(v, w, A=inner_product)
        norm_fn = lambda v: induced_norm(v, A=inner_product, axis=-1)
    else:
        inprod_fn = inner_product
        norm_fn = lambda v: np.sqrt(inprod_fn(v, v))

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod_fn(u, v)[..., None] / inprod_fn(u, u)) * u

    if normalize:
        v_orth /= norm_fn(v_orth)[..., None]

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
    all basis vectors :math:`b_i \in B` in the orthogonal basis in a numerically stable
    fashion.

    Parameters
    ----------
    v
        Vector (or stack of vectors) to orthogonalize against ``orthogonal_basis``.
    orthogonal_basis
        Orthogonal basis.
    inner_product
        Inner product defining orthogonality. Can be either a :class:`numpy.ndarray` or
        a :class:`Callable` defining the inner product. Defaults to the euclidean inner
        product.
    normalize
        Normalize the output vector, s.t. :math:`\langle v', v' \rangle = 1`.

    Returns
    -------
    v_orth :
        Orthogonalized vector.
    """
    orthogonal_basis = np.atleast_2d(orthogonal_basis)

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
        v_orth -= (inprod_fn(u, v_orth)[..., None] / inprod_fn(u, u)) * u

    if normalize:
        v_orth /= norm_fn(v_orth)[..., None]

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
    gram_schmidt_fn: Callable = modified_gram_schmidt,
) -> np.ndarray:
    r"""Perform the (modified) Gram-Schmidt process twice.

    Computes a vector :math:`v'` such that :math:`\langle v', b_i \rangle = 0` for
    all basis vectors :math:`b_i \in B` in the orthogonal basis. This performs the
    (modified) Gram-Schmidt orthogonalization process twice, which is generally more
    stable than just reorthogonalizing once. [1]_ [2]_

    Parameters
    ----------
    v
        Vector (or stack of vectors) to orthogonalize against ``orthogonal_basis``.
    orthogonal_basis
        Orthogonal basis.
    inner_product
        Inner product defining orthogonality. Can be either a :class:`numpy.ndarray` or
        a :class:`Callable` defining the inner product. Defaults to the euclidean inner
        product.
    normalize
        Normalize the output vector, s.t. :math:`\langle v', v' \rangle = 1`.
    gram_schmidt_fn
        Gram-Schmidt process to use. One of :meth:`gram_schmidt` or
        :meth:`modified_gram_schmidt`.

    Returns
    -------
    v_orth :
        Orthogonalized vector.

    References
    ----------
    .. [1] L. Giraud, J. Langou, M. Rozloznik, and J. van den Eshof, Rounding error
           analysis of the classical Gram-Schmidt orthogonalization process,
           Numer. Math., 101 (2005), pp. 87–100
    .. [2] L. Giraud, J. Langou, and M. Rozloznik, The loss of orthogonality in the
           Gram-Schmidt orthogonalization process, Comput. Math. Appl., 50 (2005)
    """
    v_orth = gram_schmidt_fn(
        v=v,
        orthogonal_basis=orthogonal_basis,
        inner_product=inner_product,
        normalize=normalize,
    )
    return gram_schmidt_fn(
        v=v_orth,
        orthogonal_basis=orthogonal_basis,
        inner_product=inner_product,
        normalize=normalize,
    )
