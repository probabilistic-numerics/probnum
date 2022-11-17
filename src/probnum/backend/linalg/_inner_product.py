"""Functions defining useful inner products and associated norms."""

from typing import Optional

from probnum.typing import MatrixType

from ... import backend as backend


def inner_product(
    x1: backend.Array,
    x2: backend.Array,
    /,
    A: Optional[MatrixType] = None,
    *,
    axis: int = -1,
) -> backend.Array:
    r"""Computes the inner product :math:`\langle x_1, x_2 \rangle_A := x_1^T A x_2` of
    two arrays along an axis.

    For n-d arrays the function computes the inner product over the given axis of the
    two arrays ``x1`` and ``x2``.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array. Must be compatible with ``x1`` for all non-contracted axes.
        The size of the axis over which to compute the inner product must be the same
        size as the respective axis in ``x1``.
    A
        Symmetric positive (semi-)definite matrix defining the geometry.
    axis
        Axis over which to compute the inner product.

    Returns
    -------
    out :
        If ``x1`` and ``x2`` are both one-dimensional arrays, a zero-dimensional array
        containing the dot product; otherwise, a non-zero-dimensional array containing
        the dot products and having rank ``N-1``, where ``N`` is the rank (number of
        dimensions) of the shape determined according to broadcasting along the
        non-contracted axes.

    Notes
    -----
    Note that the broadcasting behavior of :func:`inner_product` differs from
    :func:`numpy.inner`. Rather it follows the broadcasting rules of
    :func:`numpy.matmul` in that n-d arrays are treated as stacks of vectors.
    """
    if A is None:
        return backend.vecdot(x1, x2)

    ndim = max(x1.ndim, x2.ndim)
    x1_shape = (1,) * (ndim - x1.ndim) + tuple(x1.shape)
    x2_shape = (1,) * (ndim - x2.ndim) + tuple(x2.shape)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError("x1 and x2 must have the same shape along the given axis.")

    x1_, x2_ = backend.broadcast_arrays(x1, x2)
    x1_ = backend.move_axes(x1_, axis, -1)
    x2_ = backend.move_axes(x2_, axis, -1)

    res = x1_[..., None, :] @ (A @ x2_[..., None])
    return backend.asarray(res[..., 0, 0])


def induced_vector_norm(
    x: backend.Array,
    /,
    A: Optional[MatrixType] = None,
    axis: int = -1,
) -> backend.Array:
    r"""Induced vector norm :math:`\lVert x \rVert_A := \sqrt{x^T A x}`.

    Computes the induced norm over the given axis of the array.

    Parameters
    ----------
    x
        Array.
    A
        Symmetric positive (semi-)definite linear operator defining the geometry.
    axis
        Specifies the axis along which to compute the vector norms.

    Returns
    -------
    norm :
        Vector norm of ``x`` along the given ``axis``.
    """

    if A is None:
        return backend.linalg.vector_norm(x, ord=2, axis=axis, keepdims=False)

    x = backend.move_axes(x, axis, -1)
    y = backend.squeeze(A @ x[..., :, None], axis=-1)

    return backend.sqrt(backend.sum(x * y, axis=-1))
