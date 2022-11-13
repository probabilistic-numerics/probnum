"""Searching functions."""

from typing import Optional, Tuple

from .. import BACKEND, Array, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["argmin", "argmax", "nonzero", "where"]


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Returns the indices of the maximum values along a specified axis.

    When the maximum value occurs multiple times, only the indices corresponding to the
    first occurrence are returned.

    Parameters
    ----------
    x
        Input array. Should have a real-valued data type.
    axis
        Axis along which to search. If ``None``, the function must return the index of
        the maximum value of the flattened array.
    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array. Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result.

    Returns
    -------
    out
        If ``axis`` is ``None``, a zero-dimensional array containing the index of the
        first occurrence of the maximum value; otherwise, a non-zero-dimensional array
        containing the indices of the maximum values. The returned array must have be
        the default array index data type.
    """
    return _impl.argmax(x=x, axis=axis, keepdims=keepdims)


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Returns the indices of the minimum values along a specified axis.

    When the minimum value occurs multiple times, only the indices corresponding to the
    first occurrence are returned.

    Parameters
    ----------
    x
        Input array. Should have a real-valued data type.
    axis
        Axis along which to search. If ``None``, the function must return the index of
        the minimum value of the flattened array. Default: ``None``.
    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array. Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        If ``axis`` is ``None``, a zero-dimensional array containing the index of the
        first occurrence of the minimum value; otherwise, a non-zero-dimensional array
        containing the indices of the minimum values. The returned array must have the
        default array index data type.
    """
    return _impl.argmin(x=x, axis=axis, keepdims=keepdims)


def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """Returns the indices of the array elements which are non-zero.

    Parameters
    ----------
    x
        Input array. Must have a positive rank. If ``x`` is zero-dimensional, the
        function will raise an exception.

    Returns
    -------
    out
        A tuple of ``k`` arrays, one for each dimension of ``x`` and each of size ``n``
        (where ``n`` is the total number of non-zero elements), containing the indices
        of the non-zero elements in that dimension. The indices must be returned in
        row-major, C-style order. The returned array must have the default array index
        data type.
    """
    return _impl.nonzero(x)


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """Returns elements chosen from ``x1`` or ``x2`` depending on ``condition``.

    Parameters
    ----------
    condition
        When ``True``, yield ``x1_i``; otherwise, yield ``x2_i``. Must be compatible
        with ``x1`` and ``x2``.
    x1
        First input array. Must be compatible with ``condition`` and ``x2``.
    x2
        Second input array. Must be compatible with ``condition`` and ``x1``.

    Returns
    -------
    out
        An array with elements from ``x1`` where ``condition`` is ``True``, and elements
         from ``x2`` elsewhere. The returned array must have a data type determined by
         type promotion rules with the arrays ``x1`` and ``x2``.
    """
    return _impl.where(condition, x1, x2)
