"""Linear algebra."""
import collections
from typing import Literal, Optional, Tuple, Union

from probnum.backend.typing import ShapeLike

from .. import Array, asshape
from ..._select_backend import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_vector_norm, inner_product
from ._orthogonalize import gram_schmidt, gram_schmidt_double, gram_schmidt_modified

__all__ = [
    "cholesky",
    "cholesky_update",
    "det",
    "diagonal",
    "eigh",
    "eigvalsh",
    "einsum",
    "gram_schmidt",
    "gram_schmidt_double",
    "gram_schmidt_modified",
    "induced_vector_norm",
    "inner_product",
    "inv",
    "kron",
    "matrix_norm",
    "matrix_rank",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "solve_cholesky",
    "solve_triangular",
    "svd",
    "tensordot",
    "trace",
    "tril_to_positive_tril",
    "vecdot",
    "vector_norm",
]
__all__.sort()

cholesky = _impl.cholesky
solve_triangular = _impl.solve_triangular
solve_cholesky = _impl.solve_cholesky


def det(x: Array, /) -> Array:
    """Returns the determinant of a square matrix (or a stack of square matrices) ``x``.

    Parameters
    ----------
    x
        Input array having shape ``(..., M, M)`` and whose innermost two dimensions form
        square matrices.

    Returns
    -------
    out
        If ``x`` is a two-dimensional array, a zero-dimensional array containing the
        determinant; otherwise, a non-zero dimensional array containing the determinant
        for each square matrix.
    """
    return _impl.det(x)


def inv(x: Array, /) -> Array:
    """Returns the multiplicative inverse of a square matrix (or a stack of square
    matrices).

    Parameters
    ----------
    x
        Input array having shape ``(..., M, M)`` and whose innermost two dimensions form
        square matrices.

    Returns
    -------
    out
        An array containing the multiplicative inverses.
    """
    return _impl.inv(x)


def pinv(x: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> Array:
    """Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices).

    Parameters
    ----------
    x
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices. Should have a real-valued floating-point data type.
    rtol
        Relative tolerance for small singular values. Singular values approximately less
        than or equal to ``rtol * largest_singular_value`` are set to zero.

    Returns
    -------
    out
        An array containing the pseudo-inverses.
    """
    return _impl.pinv(x, rtol=rtol)


def matrix_rank(x: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> Array:
    """Returns the rank (i.e., number of non-zero singular values) of a matrix (or a
    stack of matrices).

    Parameters
    ----------
    x
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices. Should have a real-valued floating-point data type.
    rtol
        Relative tolerance for small singular values. Singular values approximately less
        than or equal to ``rtol * largest_singular_value`` are set to zero.

    Returns
    -------
    out
        An array containing the ranks.
    """
    return _impl.matrix_rank(x, rtol=rtol)


Slogdet = collections.namedtuple("Slogdet", ["sign", "logabsdet"])


def slogdet(x: Array, /) -> Tuple[Array, Array]:
    """Returns the sign and the natural logarithm of the absolute value of the
    determinant of a square matrix (or a stack of square matrices).

    .. note::
       The purpose of this function is to calculate the determinant more accurately when the determinant is either very small or very large, as calling ``det`` may overflow or underflow.

    Parameters
    ----------
    x
        Input array having shape ``(..., M, M)`` and whose innermost two dimensions form
        square matrices.

    Returns
    -------
    out
        A namedtuple (``sign``, ``logabsdet``) whose

        -   first element ``sign`` is an array representing the sign of the determinant
            for each square matrix.
        -   second element ``logabsdet`` is an array containing the determinant for each
            square matrix.
    """
    sign, logabsdet = _impl.slogdet(x)
    return Slogdet(sign, logabsdet)


def trace(x: Array, /, *, offset: int = 0) -> Array:
    """Returns the sum along the specified diagonals of a matrix (or a stack of
    matrices).

    Parameters
    ----------
    x
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices.
    offset
        offset specifying the off-diagonal relative to the main diagonal.
        -   ``offset = 0``: the main diagonal.
        -   ``offset > 0``: off-diagonal above the main diagonal.
        -   ``offset < 0``: off-diagonal below the main diagonal.

    Returns
    -------
    out
        An array containing the traces and whose shape is determined by removing the
        last two dimensions and storing the traces in the last array dimension.
    """
    return _impl.trace(x, offset=offset)


def einsum(
    *arrays: Array,
    optimization: Optional[str] = "greedy",
):
    """Evaluates the Einstein summation convention on the given ``arrays``.

    Using the Einstein summation convention, many common multi-dimensional, linear
    algebraic array operations can be represented in a simple fashion.

    Parameters
    ----------
    arrays
        Arrays to use for the operation.
    optimization
        Controls what kind of intermediate optimization of the contraction path should
        occur. Options are:

        +---------------+--------------------------------------------------------+
        | ``None``      | No optimization will be done.                          |
        +---------------+--------------------------------------------------------+
        | ``"optimal"`` | Exhaustively search all possible paths.                |
        +---------------+--------------------------------------------------------+
        | ``"greedy"``  | Find a path one step at a time using a cost heuristic. |
        +---------------+--------------------------------------------------------+

    Returns
    -------
    out
        The calculation based on the Einstein summation convention.
    """
    return _impl.einsum(*arrays, optimize=optimization)


def vector_norm(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal["inf", "-inf"]] = 2,
) -> Array:
    """Computes the vector norm of a vector (or batch of vectors).

    Parameters
    ----------
    x
        Input array. Should have a floating-point data type.
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
        Order of the norm. The following mathematical norms are supported:

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
        An array containing the vector norms. If ``axis`` is ``None``, the returned
        array is a zero-dimensional array containing a vector norm. If ``axis`` is a
        scalar value (``int`` or ``float``), the returned array has a rank which
        is one less than the rank of ``x``. If ``axis`` is a ``n``-tuple, the returned
        array has a rank which is ``n`` less than the rank of ``x``. The returned array
        has a floating-point data type determined by `type-promotion <https://data-apis\
        .org/array-api/latest/API_specification/type_promotion.html>`_..
    """
    return _impl.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)


def kron(x: Array, y: Array, /) -> Array:
    """Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the second array
    scaled by the first.

    Parameters
    ----------
    x
        First Kronecker factor.
    y
        Second Kronecker factor.
    """
    return _impl.kron(x, y)


def matmul(x1: Array, x2: Array, /) -> Array:
    """Computes the matrix product.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array.

    Returns
    -------
    out
        Matrix product of ``x1 and ``x2``.
    """
    return _impl.matmul(x1, x2)


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
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices. Should have a floating-point data type.
    keepdims
        If ``True``, the last two axes (dimensions) are included in the result as
        singleton dimensions, and, accordingly, the result is compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`_). Otherwise, if ``False``, the last two
        axes (dimensions) are not be included in the result.
    ord
        Order of the norm. The following mathematical norms are supported:

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
        An array containing the norms for each ``MxN`` matrix. If ``keepdims`` is
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
        Coefficient array ``A`` having shape ``(..., M, M)`` and whose innermost two
        dimensions form square matrices. Must be of full rank (i.e., all rows or,
        equivalently, columns must be linearly independent).
    x2
        Ordinate (or "dependent variable") array ``B``. If ``x2`` has shape ``(M,)``,
        ``x2`` is equivalent to an array having shape ``(..., M, 1)``. If ``x2`` has
        shape ``(..., M, K)``, each column ``k`` defines a set of ordinate values for
        which to compute a solution, and ``shape(x2)[:-1]`` must be compatible with
        ``shape(x1)[:-1]`` (see `broadcasting <https://data-apis.org/array-api/latest\
        /API_specification/broadcasting.html>`_).

    Returns
    -------
    out:
        An array containing the solution to the system ``AX = B`` for each square
        matrix. The returned array must have the same shape as ``x2`` (i.e., the array
        corresponding to ``B``) and must have a floating-point data type determined by
        `type-promotion <https://data-apis.org/array-api/latest/API_specification\
        /type_promotion.html>`_.
    """
    return _impl.solve(x1, x2)


def diagonal(
    x: Array, /, *, offset: int = 0, axis1: int = -2, axis2: int = -1
) -> Array:
    """Returns the specified diagonals of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions
        form ``MxN`` matrices.
    offset
        Offset specifying the off-diagonal relative to the main diagonal.
        - ``offset = 0``: the main diagonal.
        - ``offset > 0``: off-diagonal above the main diagonal.
        - ``offset < 0``: off-diagonal below the main diagonal.
    axis1
        Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals
        should be taken.
    axis2
        Axis to be used as the second axis of the 2-D sub-arrays from which the
        diagonals should be taken.

    Returns
    -------
    out
        An array containing the diagonals and whose shape is determined by removing the
        last two dimensions and appending a dimension equal to the size of the resulting
        diagonals.
    """
    return _impl.diagonal(x, offset, axis1, axis2)


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
        Input array having shape ``(..., M, M)`` and whose innermost two dimensions form
        square matrices. Must have a floating-point data type.

    Returns
    -------
    out
        A namedtuple (``eigenvalues``, ``eigenvectors``) whose

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


def eigvalsh(x: Array, /) -> Array:
    """Returns the eigenvalues of a symmetric matrix (or a stack of symmetric matrices).

    Parameters
    ----------
    x
        Input array having shape ``(..., M, M)`` and whose innermost two dimensions form
         square matrices. Must have a real-valued floating-point data type.

    Returns
    -------
    out
        An array containing the computed eigenvalues. The returned array must have shape
         ``(..., M)`` and have the same data type as ``x``.

    .. note::
       Eigenvalue sort order is left unspecified and is thus implementation-dependent.
    """
    return _impl.eigvalsh(x)


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
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
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
        A namedtuple ``(U, S, Vh)`` whose

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
    x: Array, /, *, mode: Literal["reduced", "complete", "r"] = "reduced"
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
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices of rank ``N``. Should have a floating-point data type.
    mode
        Decomposition mode. Should be one of the following modes:

        -   ``'reduced'``: compute only the leading ``K`` columns of ``q``, such that
            ``q`` and ``r`` have dimensions ``(..., M, K)`` and ``(..., K, N)``,
            respectively, and where ``K = min(M, N)``.
        -   ``'complete'``: compute ``q`` and ``r`` with dimensions ``(..., M, M)`` and
            ``(..., M, N)``, respectively.

    Returns
    -------
    out
        A namedtuple ``(Q, R)`` whose

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


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    """Computes the (vector) dot product of two arrays along an axis.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array. Must be compatible with ``x1`` for all non-contracted axes.
        The size of the axis over which to compute the dot product must be the same size
        as the respective axis in ``x1``.
    axis
        Axis over which to compute the dot product.

    Returns
    -------
    out
        If ``x1`` and ``x2`` are both one-dimensional arrays, a zero-dimensional array
        containing the dot product; otherwise, a non-zero-dimensional array containing
        the dot products and having rank ``N-1``, where ``N`` is the rank (number of
        dimensions) of the shape determined according to broadcasting along the
        non-contracted axes.
    """
    return _impl.vecdot(x1, x2, axis)


def tensordot(
    x1: Array, x2: Array, /, *, axes: Union[int, Tuple[ShapeLike, ShapeLike]] = 2
) -> Array:
    """Returns a tensor contraction of ``x1`` and ``x2`` over specific axes.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array. Corresponding contracted axes of ``x1`` and ``x2`` must be equal.
    axes
        Number of axes (dimensions) to contract or explicit sequences of axes
        (dimensions) for ``x1`` and ``x2``, respectively.

        If ``axes`` is an ``int`` equal to ``N``, then contraction will be performed
        over the last ``N`` axes of ``x1`` and the first ``N`` axes of ``x2`` in order.
        The size of each corresponding axis (dimension) must match.
        -   If ``N`` equals ``0``, the result is the tensor (outer) product.
        -   If ``N`` equals ``1``, the result is the tensor dot product.
        -   If ``N`` equals ``2``, the result is the tensor double contraction (default).

        If ``axes`` is a tuple of two sequences ``(x1_axes, x2_axes)``, the first
        sequence must apply to ``x`` and the second sequence to ``x2``. Both sequences
        must have the same length. Each axis (dimension) ``x1_axes[i]`` for ``x1`` must
        have the same size as the respective axis (dimension) ``x2_axes[i]`` for ``x2``.
        Each sequence must consist of unique (nonnegative) integers that specify valid
        axes for each respective array.

    Returns
    -------
    out
        An array containing the tensor contraction whose shape consists of the
        non-contracted axes (dimensions) of the first array ``x1``, followed by the
        non-contracted axes (dimensions) of the second array ``x2``.
    """
    if isinstance(axes, tuple):
        axes = (asshape(axes[0]), asshape(axes[1]))
    return _impl.tensordot(x1, x2, axes)
