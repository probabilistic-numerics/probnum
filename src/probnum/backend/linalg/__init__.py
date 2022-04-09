"""Linear algebra."""

from typing import Literal, Optional, Tuple, Union

from .. import BACKEND, Array, Backend

__all__ = [
    "vector_norm",
    "matrix_norm",
    "induced_norm",
    "inner_product",
    "gram_schmidt",
    "modified_gram_schmidt",
    "double_gram_schmidt",
    "cholesky",
    "solve",
    "solve_triangular",
    "solve_cholesky",
    "cholesky_update",
    "tril_to_positive_tril",
    "qr",
    "svd",
    "eigh",
]

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_norm, inner_product
from ._orthogonalize import double_gram_schmidt, gram_schmidt, modified_gram_schmidt

cholesky = _impl.cholesky
solve_triangular = _impl.solve_triangular
solve_cholesky = _impl.solve_cholesky
qr = _impl.qr
svd = _impl.svd
eigh = _impl.eigh


def vector_norm(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal["inf", "-inf"]] = 2,
) -> Array:
    """Computes the vector norm of a vector (or batch of vectors) ``x``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.

    axis
        If an integer, ``axis`` specifies the axis (dimension) along which to compute
        vector norms. If an n-tuple, ``axis`` specifies the axes (dimensions) along
        which to compute batched vector norms. If ``None``, the vector norm is
        computed over all array values (i.e., equivalent to computing the vector norm of
        a flattened array).
    keepdims
        If ``True``, the axes (dimensions) specified by ``axis`` are included in the
        result as singleton dimensions, and, accordingly, the result is compatible with
        the input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`_). Otherwise, if ``False``, the last two
        axes (dimensions) are not be included in the result.
    ord
        order of the norm. The following mathematical norms are supported:

        +------------------+----------------------------+
        | ord              | description                |
        +==================+============================+
        | 1                | L1-norm (Manhattan)        |
        +------------------+----------------------------+
        | 2                | L2-norm (Euclidean)        |
        +------------------+----------------------------+
        | inf              | infinity norm              |
        +------------------+----------------------------+
        | (int,float >= 1) | p-norm                     |
        +------------------+----------------------------+

        The following non-mathematical "norms" are supported:

        +------------------+--------------------------------+
        | ord              | description                    |
        +==================+================================+
        | 0                | sum(a != 0)                    |
        +------------------+--------------------------------+
        | -1               | 1./sum(1./abs(a))              |
        +------------------+--------------------------------+
        | -2               | 1./sqrt(sum(1./abs(a)\*\*2))   |
        +------------------+--------------------------------+
        | -inf             | min(abs(a))                    |
        +------------------+--------------------------------+
        | (int,float < 1)  | sum(abs(a)\*\*ord)\*\*(1./ord) |
        +------------------+--------------------------------+

    Returns
    -------
    out
        an array containing the vector norms. If ``axis`` is ``None``, the returned
        array is a zero-dimensional array containing a vector norm. If ``axis`` is a
        scalar value (``int`` or ``float``), the returned array has a rank which
        is one less than the rank of ``x``. If ``axis`` is a ``n``-tuple, the returned
        array has a rank which is ``n`` less than the rank of ``x``. The returned array
        has a floating-point data type determined by `type-promotion <https://data-apis\
        .org/array-api/latest/API_specification/type_promotion.html>`_..
    """
    return _impl.vector_norm(x=x, axis=axis, keepdims=keepdims, ord=ord)


def matrix_norm(
    x: Array,
    /,
    *,
    keepdims: bool = False,
    ord: Optional[Union[int, float, Literal["inf", "-inf", "fro", "nuc"]]] = "fro",
) -> Array:
    """Computes the matrix norm of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices. Should have a floating-point data type.
    keepdims
        If ``True``, the last two axes (dimensions) are included in the result as
        singleton dimensions, and, accordingly, the result is compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`_). Otherwise, if ``False``, the last two
        axes (dimensions) are not be included in the result.
    ord
        order of the norm. The following mathematical norms are supported:

        +------------------+---------------------------------+
        | ord              | description                     |
        +==================+=================================+
        | 'fro'            | Frobenius norm                  |
        +------------------+---------------------------------+
        | 'nuc'            | nuclear norm                    |
        +------------------+---------------------------------+
        | 1                | max(sum(abs(x), axis=0))        |
        +------------------+---------------------------------+
        | 2                | largest singular value          |
        +------------------+---------------------------------+
        | inf              | max(sum(abs(x), axis=1))        |
        +------------------+---------------------------------+

        The following non-mathematical "norms" are supported:

        +------------------+---------------------------------+
        | ord              | description                     |
        +==================+=================================+
        | -1               | min(sum(abs(x), axis=0))        |
        +------------------+---------------------------------+
        | -2               | smallest singular value         |
        +------------------+---------------------------------+
        | -inf             | min(sum(abs(x), axis=1))        |
        +------------------+---------------------------------+

        If ``ord=1``, the norm corresponds to the induced matrix norm where ``p=1``
        (i.e., the maximum absolute value column sum).
        If ``ord=2``, the norm corresponds to the induced matrix norm where ``p=inf``
        (i.e., the maximum absolute value row sum).
        If ``ord=inf``, the norm corresponds to the induced matrix norm where ``p=2``
        (i.e., the largest singular value).

    Returns
    -------
    out
        an array containing the norms for each ``MxN`` matrix. If ``keepdims`` is
        ``False``, the returned array has a rank which is two less than the
        rank of ``x``. The returned array must have a floating-point data type
        determined by `type-promotion <https://data-apis.org/array-api/latest/\
        API_specification/type_promotion.html>`_.
    """
    return _impl.matrix_norm(x=x, keepdims=keepdims, ord=ord)


def solve(x1: Array, x2: Array, /) -> Array:
    """Returns the solution to the system of linear equations represented by the
    well-determined (i.e., full rank) linear matrix equation ``AX = B``.

    .. note::

       Whether an array library explicitly checks whether an input array is full rank is
       implementation-defined.

    Parameters
    ----------
    x1
        coefficient array ``A`` having shape ``(..., M, M)`` and whose innermost two
        dimensions form square matrices. Must be of full rank (i.e., all rows or,
        equivalently, columns must be linearly independent). Should have a
        floating-point data type.
    x2
        ordinate (or "dependent variable") array ``B``. If ``x2`` has shape ``(M,)``,
        ``x2`` is equivalent to an array having shape ``(..., M, 1)``. If ``x2`` has
        shape ``(..., M, K)``, each column ``k`` defines a set of ordinate values for
        which to compute a solution, and ``shape(x2)[:-1]`` must be compatible with
        ``shape(x1)[:-1]`` (see `broadcasting <https://data-apis.org/array-api/latest\
        /API_specification/broadcasting.html>`_). Should have a floating-point data
        type.

    Returns
    -------
    out:
        an array containing the solution to the system ``AX = B`` for each square
        matrix. The returned array must have the same shape as ``x2`` (i.e., the array
        corresponding to ``B``) and must have a floating-point data type determined by
        `type-promotion <https://data-apis.org/array-api/latest/API_specification\
        /type_promotion.html>`_.
    """
    return _impl.solve(x1, x2)
