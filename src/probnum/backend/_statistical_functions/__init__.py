"""Statistical functions."""

from __future__ import annotations

from typing import Optional, Tuple, Union

from .. import Array, DType
from ..._select_backend import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["max", "mean", "min", "prod", "std", "sum", "var"]
__all__.sort()


def max(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """Calculates the maximum value of the input array ``x``.

    **Special Cases**
    For floating-point operands,

    - If ``x_i`` is ``NaN``, the maximum value is ``NaN`` (i.e., ``NaN`` values
      propagate).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which maximum values must be computed. By default, the
        maximum value must be computed over the entire array. If a tuple of integers,
        maximum values must be computed over multiple axes. Default: ``None``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the maximum value was computed over the entire array, a zero-dimensional
        array containing the maximum value; otherwise, a non-zero-dimensional array
        containing the maximum values. The returned array must have the same data type
        as ``x``.
    """
    return _impl.max(x, axis=axis, keepdims=keepdims)


def mean(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """Calculates the arithmetic mean of the input array ``x``.

    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the arithmetic mean.

    - If ``N`` is ``0``, the arithmetic mean is ``NaN``.
    - If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN`` (i.e., ``NaN`` values
      propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which arithmetic means must be computed. By default, the mean
        must be computed over the entire array. If a tuple of integers, arithmetic means
        must be computed over multiple axes. Default: ``None``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the arithmetic mean was computed over the entire array, a zero-dimensional
        array containing the arithmetic mean; otherwise, a non-zero-dimensional array
        containing the arithmetic means. The returned array must have the same data type
        as ``x``.
    """
    return _impl.mean(x, axis=axis, keepdims=keepdims)


def min(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """Calculates the minimum value of the input array ``x``.

    **Special Cases**
    For floating-point operands,

    - If ``x_i`` is ``NaN``, the minimum value is ``NaN`` (i.e., ``NaN`` values
      propagate).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which minimum values must be computed. By default, the
        minimum value must be computed over the entire array. If a tuple of integers,
        minimum values must be computed over multiple axes. Default: ``None``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the minimum value was computed over the entire array, a zero-dimensional
        array containing the minimum value; otherwise, a non-zero-dimensional array
        containing the minimum values. The returned array must have the same data type
        as ``x``.
    """
    return _impl.min(x, axis=axis, keepdims=keepdims)


def prod(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[DType] = None,
    keepdims: bool = False,
) -> Array:
    """Calculates the product of input array ``x`` elements.

    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the product.

    - If ``N`` is ``0``, the product is `1` (i.e., the empty product).

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the product is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which products must be computed. By default, the product must
        be computed over the entire array. If a tuple of integers, products must be
        computed over multiple axes. Default: ``None``.
    dtype
        data type of the returned array.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the product was computed over the entire array, a zero-dimensional array
        containing the product; otherwise, a non-zero-dimensional array containing the
        products. The returned array must have a data type as described by the ``dtype``
        parameter above.
    """
    return _impl.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    """Calculates the standard deviation of the input array ``x``.

    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the standard deviation.

    - If ``N - correction`` is less than or equal to ``0``, the standard deviation is
      ``NaN``.
    - If ``x_i`` is ``NaN``, the standard deviation is ``NaN`` (i.e., ``NaN`` values
      propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which standard deviations must be computed. By default, the
        standard deviation must be computed over the entire array. If a tuple of
        integers, standard deviations must be computed over multiple axes.
        Default: ``None``.
    correction
        degrees of freedom adjustment. Setting this parameter to a value other than
        ``0`` has the effect of adjusting the divisor during the calculation of the
        standard deviation according to ``N-c`` where ``N`` corresponds to the total
        number of elements over which the standard deviation is computed and ``c``
        corresponds to the provided degrees of freedom adjustment. When computing the
        standard deviation of a population, setting this parameter to ``0`` is the
        standard choice (i.e., the provided array contains data constituting an entire
        population). When computing the corrected sample standard deviation, setting
        this parameter to ``1`` is the standard choice (i.e., the provided array
        contains data sampled from a larger population; this is commonly referred to as
        Bessel's correction). Default: ``0``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the standard deviation was computed over the entire array, a zero-dimensional
        array containing the standard deviation; otherwise, a non-zero-dimensional array
        containing the standard deviations. The returned array must have the same data
        type as ``x``.
    """
    return _impl.std(x, axis=axis, correction=correction, keepdims=keepdims)


def sum(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[DType] = None,
    keepdims: bool = False,
) -> Array:
    """Calculates the sum of the input array ``x``.

    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the sum.

    - If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the sum is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which sums must be computed. By default, the sum must be
        computed over the entire array. If a tuple of integers, sums must be computed
        over multiple axes. Default: ``None``.
    dtype
        data type of the returned array.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the sum was computed over the entire array, a zero-dimensional array
        containing the sum; otherwise, an array containing the sums. The returned
        array must have a data type as described by the ``dtype`` parameter above.
    """
    return _impl.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


def var(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    """Calculates the variance of the input array ``x``.

    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the variance.

    - If ``N - correction`` is less than or equal to ``0``, the variance is ``NaN``.
    - If ``x_i`` is ``NaN``, the variance is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which variances must be computed. By default, the variance
        must be computed over the entire array. If a tuple of integers, variances must
        be computed over multiple axes. Default: ``None``.
    correction
        degrees of freedom adjustment. Setting this parameter to a value other than
        ``0`` has the effect of adjusting the divisor during the calculation of the
        variance according to ``N-c`` where ``N`` corresponds to the total number of
        elements over which the variance is computed and ``c`` corresponds to the
        provided degrees of freedom adjustment. When computing the variance of a
        population, setting this parameter to ``0`` is the standard choice (i.e., the
        provided array contains data constituting an entire population). When computing
        the unbiased sample variance, setting this parameter to ``1`` is the standard
        choice (i.e., the provided array contains data sampled from a larger population;
        this is commonly referred to as Bessel's correction). Default: ``0``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see `broadcasting <https://data-apis.org/array-api/latest/\
        API_specification/broadcasting.html>`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out
        if the variance was computed over the entire array, a zero-dimensional array
        containing the variance; otherwise, a non-zero-dimensional array containing the
        variances. The returned array must have the same data type as ``x``.
    """
    return _impl.var(x, axis=axis, correction=correction, keepdims=keepdims)
