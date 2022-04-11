"""Linear algebra."""
import collections
from typing import Literal, Optional, Tuple, Union

from .. import BACKEND, Array, Backend

__all__ = [
    "cholesky",
    "cholesky_update",
    "eigh",
    "gram_schmidt",
    "gram_schmidt_double",
    "gram_schmidt_modified",
    "induced_norm",
    "inner_product",
    "matrix_norm",
    "qr",
    "solve",
    "solve_cholesky",
    "solve_triangular",
    "svd",
    "tril_to_positive_tril",
    "vector_norm",
]

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_norm, inner_product
from ._orthogonalize import gram_schmidt, gram_schmidt_double, gram_schmidt_modified

cholesky = _impl.cholesky
solve_triangular = _impl.solve_triangular
solve_cholesky = _impl.solve_cholesky


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
    return _impl.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)


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
    return _impl.matrix_norm(x, keepdims=keepdims, ord=ord)


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


Eigh = collections.namedtuple("Eigh", ["eigenvalues", "eigenvectors"])


def eigh(x: Array, /) -> Tuple[Array]:
    """
    Returns an eigendecomposition ``x = QLQᵀ`` of a symmetric matrix (or a stack of
    symmetric matrices) ``x``, where ``Q`` is an orthogonal matrix (or a stack of
    matrices) and ``L`` is a vector (or a stack of vectors).

    .. note::

       Whether an array library explicitly checks whether an input array is a symmetric
       matrix (or a stack of symmetric matrices) is implementation-defined.

    Parameters
    ----------
    x
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form
        square matrices. Must have a floating-point data type.

    Returns
    -------
    out
        a namedtuple (``eigenvalues``, ``eigenvectors``) whose

        -   first element is an array consisting of computed eigenvalues and has shape
            ``(..., M)``.
        -   second element is an array where the columns of the inner most
            matrices contain the computed eigenvectors. These matrices are
            orthogonal. The array containing the eigenvectors has shape
            ``(..., M, M)``.

        Each returned array has the same floating-point data type as ``x``.

    .. note::

       Eigenvalue sort order is left unspecified and is thus implementation-dependent.
    """
    eigenvalues, eigenvectors = _impl.eigh(x)
    return Eigh(eigenvalues, eigenvectors)


SVD = collections.namedtuple("SVD", ["U", "S", "Vh"])


def svd(x: Array, /, *, full_matrices: bool = True) -> Union[Array, Tuple[Array, ...]]:
    """
    Returns a singular value decomposition ``A = USVh`` of a matrix (or a stack of
    matrices) ``x``, where ``U`` is a matrix (or a stack of matrices) with orthonormal
    columns, ``S`` is a vector of non-negative numbers (or stack of vectors), and ``Vh``
    is a matrix (or a stack of matrices) with orthonormal rows.

    Parameters
    ----------
    x
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        matrices on which to perform singular value decomposition. Must have a
        floating-point data type.
    full_matrices
        If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has shape
        ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``, compute on
        the leading ``K`` singular vectors, such that ``U`` has shape ``(..., M, K)``
        and ``Vh`` has shape ``(..., K, N)`` and where ``K = min(M, N)``.

    Returns
    -------
    out
        a namedtuple ``(U, S, Vh)`` whose

        -   first element is an array whose shape depends on the value of
            ``full_matrices`` and contains matrices with orthonormal columns (i.e., the
            columns are left singular vectors). If
            ``full_matrices`` is ``True``, the array has shape ``(..., M, M)``. If
            ``full_matrices`` is ``False``, the array has shape ``(..., M, K)``,
            where ``K = min(M, N)``. The first ``x.ndim-2`` dimensions have the
            same shape as those of the input ``x``.
        -   second element is an array with shape ``(..., K)`` that contains the
            vector(s) of singular values of length ``K``, where ``K = min(M, N)``. For
            each vector, the singular values must be sorted in descending order by
            magnitude, such that ``s[..., 0]`` is the
            largest value, ``s[..., 1]`` is the second largest value, et cetera. The
            first ``x.ndim-2`` dimensions have the same shape as those of the input
            ``x``.
        -   third element is an array whose shape depends on the value of
            ``full_matrices`` and contain orthonormal rows (i.e., the rows are the right
            singular vectors and the array is the adjoint). If ``full_matrices`` is
            ``True``, the array has shape ``(..., N, N)``. If ``full_matrices`` is
            ``False``, the array has shape ``(..., K, N)`` where ``K = min(M, N)``.
            The first ``x.ndim-2`` dimensions have the same shape as those of the input
            ``x``.

        Each returned array has the same floating-point data type as ``x``.
    """
    U, S, Vh = _impl.svd(x, full_matrices=full_matrices)
    return SVD(U, S, Vh)


QR = collections.namedtuple("QR", ["Q", "R"])


def qr(
    x: Array, /, *, mode: Literal["reduced", "complete"] = "reduced"
) -> Tuple[Array, Array]:
    """
    Returns the QR decomposition ``x = QR`` of a full column rank matrix (or a stack of
    matrices), where ``Q`` is an orthonormal matrix (or a stack of matrices) and ``R``
    is an upper-triangular matrix (or a stack of matrices).

    .. note::

       Whether an array library explicitly checks whether an input array is a full
       column rank matrix (or a stack of full column rank matrices) is
       implementation-defined.

    Parameters
    ----------
    x
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices of rank ``N``. Should have a floating-point data type.
    mode
        decomposition mode. Should be one of the following modes:

        -   ``'reduced'``: compute only the leading ``K`` columns of ``q``, such that
            ``q`` and ``r`` have dimensions ``(..., M, K)`` and ``(..., K, N)``,
            respectively, and where ``K = min(M, N)``.
        -   ``'complete'``: compute ``q`` and ``r`` with dimensions ``(..., M, M)`` and
            ``(..., M, N)``, respectively.

    Returns
    -------
    out
        a namedtuple ``(Q, R)`` whose

        -   first element is an array whose shape depends on the value of ``mode`` and
            contains matrices with orthonormal columns. If ``mode`` is ``'complete'``,
            the array has shape ``(..., M, M)``. If ``mode`` is ``'reduced'``, the array
            has shape ``(..., M, K)``, where ``K = min(M, N)``. The first ``x.ndim-2``
            dimensions have the same size as those of the input array ``x``.
        -   second element is an array whose shape depends on the value of ``mode`` and
            contains upper-triangular matrices. If ``mode`` is ``'complete'``, the array
            has shape ``(..., M, N)``. If ``mode`` is ``'reduced'``, the array has shape
            ``(..., K, N)``, where ``K = min(M, N)``. The first ``x.ndim-2`` dimensions
            have the same size as those of the input ``x``.

        Each returned array has a floating-point data type determined by
        `type-promotion <https://data-apis.org/array-api/latest/API_specification/\
        type_promotion.html>`_.
    """
    Q, R = _impl.qr(x, mode=mode)
    return QR(Q, R)
