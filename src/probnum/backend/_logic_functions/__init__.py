"""Logic functions."""

from .. import BACKEND, Array, Backend
from ..typing import ShapeType

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

from typing import Optional, Union

__all__ = [
    "all",
    "any",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
]
__all__.sort()


def all(
    x: Array, /, *, axis: Optional[Union[int, ShapeType]] = None, keepdims: bool = False
) -> Array:
    """Tests whether all input array elements evaluate to ``True`` along a specified
    axis.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis or axes along which to perform a logical ``AND`` reduction. By default, the
        logical ``AND`` reduction will be performed over the entire array.
    keepdims
        If ``True``, the reduced axes (dimensions) will be included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes (dimensions)
        will not be included in the result.

    Returns
    -------
    out
        If a logical ``AND`` reduction was performed over the entire array, the returned
        array will be a zero-dimensional array containing the test result; otherwise,
        the returned array will be a non-zero-dimensional array containing the test
        results.
    """
    return _impl.all(x, axis=axis, keepdims=keepdims)


def any(
    x: Array, /, *, axis: Optional[Union[int, ShapeType]] = None, keepdims: bool = False
) -> Array:
    """Tests whether any input array element evaluates to ``True`` along a specified
    axis.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis or axes along which to perform a logical ``OR`` reduction. By default, the
        logical ``OR`` reduction will be performed over the entire array.
    keepdims
        If ``True``, the reduced axes (dimensions) will be included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes (dimensions)
        will not be included in the result.

    Returns
    -------
    out
        If a logical ``OR`` reduction was performed over the entire array, the returned
        array will be a zero-dimensional array containing the test result; otherwise,
        the returned array will be a non-zero-dimensional array containing the test
        results.
    """
    return _impl.any(x, axis=axis, keepdims=keepdims)


def equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with ``x1``. May have any data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.equal(x1, x2)


def greater(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i > x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.greater(x1, x2)


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i >= x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.greater_equal(x1, x2)


def less(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.less(x1, x2)


def less_equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i <= x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.less_equal(x1, x2)


def logical_and(x1: Array, x2: Array, /) -> Array:
    """Computes the logical AND for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a boolean data
        type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_and(x1, x2)


def logical_not(x: Array, /) -> Array:
    """Computes the logical NOT for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_not(x)


def logical_or(x1: Array, x2: Array, /) -> Array:
    """Computes the logical OR for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_or(x1, x2)


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """Computes the logical XOR for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_xor(x1, x2)


def not_equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with ``x1``.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.not_equal(x1, x2)
