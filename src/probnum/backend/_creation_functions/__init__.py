"""Array creation functions."""

from __future__ import annotations

from typing import List, Optional, Union

from .. import BACKEND, Array, Backend, Device, Dtype, Scalar, ndim
from ..typing import DTypeLike, ScalarLike, ShapeType

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = [
    "arange",
    "asarray",
    "asscalar",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]


def asarray(
    obj: Union[Array, bool, int, float, "NestedSequence", "SupportsBufferProtocol"],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    """Convert the input to an array.

    Parameters
    ----------
    obj
        object to be converted to an array. May be a Python scalar, a (possibly nested)
        sequence of Python scalars, or an object supporting the Python buffer protocol.

        .. admonition:: Tip
           :class: important

           An object supporting the buffer protocol can be turned into a memoryview
           through ``memoryview(obj)``.

    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from the data type(s) in ``obj``. If all input values are
        Python scalars, then

        -   if all values are of type ``bool``, the output data type must be ``bool``.
        -   if the values are a mixture of ``bool``\s and ``int``, the output data
            type must be the default integer data type.
        -   if one or more values are ``float``\s, the output data type must be the
            default floating-point data type.

        Default: ``None``.

        .. admonition:: Note
           :class: note

           If ``dtype`` is not ``None``, then array conversions should obey
           `type-promotion <https://data-apis.org/array-api/latest/API_specification\
           /type_promotion.html>`_ rules. Conversions not specified according to
           `type-promotion <https://data-apis.org/array-api/latest/API_specification\
           /type_promotion.html>`_ rules may or may not be permitted by a conforming
           array library. To perform an explicit cast, use :func:`astype`.

    device
        device on which to place the created array. If ``device`` is ``None`` and ``x``
        is an array, the output array device must be inferred from ``x``. Default:
        ``None``.
    copy
        boolean indicating whether or not to copy the input. If ``True``, the function
        must always copy. If ``False``, the function must never copy for input which
        supports the buffer protocol and must raise a ``ValueError`` in case a copy
        would be necessary. If ``None``, the function must reuse existing memory buffer
        if possible and copy otherwise. Default: ``None``.

    Returns
    -------
    out
        an array containing the data from ``obj``.
    """
    return _impl.asarray(obj, dtype=dtype, device=device, copy=copy)


def asscalar(x: ScalarLike, dtype: DTypeLike = None) -> Scalar:
    """Convert a scalar into a NumPy scalar.

    Parameters
    ----------
    x
        Scalar value.
    dtype
        Data type of the scalar.
    """
    if ndim(x) != 0:
        raise ValueError("The given input is not a scalar.")

    return asarray(x, dtype=dtype)[()]


def tril(x: Array, /, *, k: int = 0) -> Array:
    """Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::

       The lower triangular part of the matrix is defined as the elements on and below
       the specified diagonal ``k``.

    Parameters
    ----------
    x
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices.
    k
        diagonal above which to zero elements. If ``k = 0``, the diagonal is the main
        diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``,
        the diagonal is above the main diagonal. Default: ``0``.

        .. note::

           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on
           the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out :
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as ``x``. All elements above the specified diagonal
        ``k`` must be zeroed. The returned array should be allocated on the same device
        as ``x``.
    """
    return _impl.tril(x, k=k)


def triu(x: Array, /, *, k: int = 0) -> Array:
    """Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::

       The upper triangular part of the matrix is defined as the elements on and above
       the specified diagonal ``k``.

    Parameters
    ----------
    x
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices.
    k
        Diagonal below which to zero elements. If ``k = 0``, the diagonal is the main
        diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``,
        the diagonal is above the main diagonal. Default: ``0``.

        .. note::

           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on
           the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out:
        An array containing the upper triangular part(s). The returned array must have
        the same shape and data type as ``x``. All elements below the specified diagonal
        ``k`` must be zeroed. The returned array should be allocated on the same device
        as ``x``.
    """
    return _impl.triu(x, k=k)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns evenly spaced values within the half-open interval ``[start, stop)`` as a
    one-dimensional array.

    Parameters
    ----------
    start
        if ``stop`` is specified, the start of interval (inclusive); otherwise, the end
        of the interval (exclusive). If ``stop`` is not specified, the default starting
        value is ``0``.
    stop
        the end of the interval. Default: ``None``.
    step
        the distance between two adjacent elements (``out[i+1] - out[i]``). Must not be
        ``0``; may be negative, this results in an empty array if ``stop >= start``.
        Default: ``1``.
    dtype
        output array data type. Should be a floating-point data type. If ``dtype`` is
        ``None``, the output array data type must be the default floating-point data
        type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.

    .. note::

       This function cannot guarantee that the interval does not include the ``stop``
       value in those cases where ``step`` is not an integer and floating-point rounding
       errors affect the length of the output array.

    Returns
    -------
    out
        a one-dimensional array containing evenly spaced values. The length of the
        output array must be ``ceil((stop-start)/step)`` if ``stop - start`` and
        ``step`` have the same sign, and length ``0`` otherwise.
    """
    return _impl.arange(start, stop, step, dtype=dtype, device=device)


def empty(
    shape: ShapeType,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns an uninitialized array having a specified ``shape``.

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out
        an array containing uninitialized data.
    """
    return _impl.empty(shape, dtype=dtype, device=device)


def empty_like(
    x: Array,
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns an uninitialized array with the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out
        an array having the same shape as ``x`` and containing uninitialized data.
    """
    return _impl.empty_like(x, dtype=dtype, device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a two-dimensional array with ones on the ``k``\\ th diagonal and zeros
    elsewhere.

    Parameters
    ----------
    n_rows
        number of rows in the output array.
    n_cols
        number of columns in the output array. If ``None``, the default number of
        columns in the output array is equal to ``n_rows``. Default: ``None``.
    k
        index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and ``0`` to the main diagonal. Default: ``0``.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out
        an array where all elements are equal to zero, except for the ``k``\\th
        diagonal, whose values are equal to one.
    """
    return _impl.eye(n_rows, n_cols, k=k, dtype=dtype, device=device)


def full(
    shape: ShapeType,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape
        output array shape.
    fill_value
        fill value.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``fill_value``. If the fill value is an ``int``, the
        output array data type must be the default integer data type. If the fill value
        is a ``float``, the output array data type must be the default floating-point
        data type. If the fill value is a ``bool``, the output array must have boolean
        data type. Default: ``None``.

        .. note::

           If the ``fill_value`` exceeds the precision of the resolved default output
           array data type, behavior is left unspecified and, thus,
           implementation-defined.

    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out
        an array where every element is equal to ``fill_value``.
    """
    return _impl.full(shape, fill_value, dtype=dtype, device=device)


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a new array filled with ``fill_value`` and having the same ``shape`` as
    an input array ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    fill_value
        fill value.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.

        .. note::

           If the ``fill_value`` exceeds the precision of the resolved output array data
           type, behavior is unspecified and, thus, implementation-defined.

        .. note::

           If the ``fill_value`` has a data type (``int`` or ``float``) which is not of
           the same data type kind as the resolved output array data type, behavior is
           unspecified and, thus, implementation-defined.

    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out
        an array having the same shape as ``x`` and where every element is equal to
        ``fill_value``.
    """
    return _impl.full_like(x, fill_value=fill_value, dtype=dtype, device=device)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    """Returns evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start
        the start of the interval.
    stop
        the end of the interval. If ``endpoint`` is ``False``, the function must
        generate a sequence of ``num+1`` evenly spaced numbers starting with ``start``
        and ending with ``stop`` and exclude the ``stop`` from the returned array such
        that the returned array consists of evenly spaced numbers over the half-open
        interval ``[start, stop)``. If ``endpoint`` is ``True``, the output array must
        consist of evenly spaced numbers over the closed interval ``[start, stop]``.
        Default: ``True``.

        .. note::

           The step size changes when `endpoint` is `False`.

    num
        number of samples. Must be a non-negative integer value; otherwise, the function
        must raise an exception.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    endpoint
        boolean indicating whether to include ``stop`` in the interval. Default:
        ``True``.

    Returns
    -------
    out
        a one-dimensional array containing evenly spaced values.
    """
    return _impl.linspace(
        start, stop, num=num, dtype=dtype, device=device, endpoint=endpoint
    )


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    """Returns coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays
        an arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array should have the same numeric data type.
    indexing
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or
        one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
        respectively), the ``indexing`` keyword has no effect and should be ignored.
        Default: ``'xy'``.

    Returns
    -------
    out
        list of N arrays, where ``N`` is the number of provided one-dimensional input
        arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional
        arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array must have the shape
          ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape
          ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of
        length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must
        have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned
        array must have shape ``(N, M)``.
        Similarly, for the three-dimensional case with input one-dimensional arrays of
        length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned
        array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then
        each returned array must have shape ``(N, M, P)``.
        Each returned array should have the same data type as the input arrays.
    """
    return _impl.ones_like(*arrays, indexing=indexing)


def ones(
    shape: ShapeType,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    Returns
    -------
    out
        an array containing ones.
    """
    return _impl.ones(shape, dtype=dtype, device=device)


def ones_like(
    x: Array,
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a new array filled with ones and having the same ``shape`` as an input
    array ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out
        an array having the same shape as ``x`` and filled with ones.
    """
    return _impl.ones_like(x, dtype=dtype, device=device)


def zeros(
    shape: ShapeType,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out
        an array containing zeros.
    """
    return _impl.zeros(shape, dtype=dtype, device=device)


def zeros_like(
    x: Array,
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """Returns a new array filled with zeros and having the same ``shape`` as an input
    array ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    out
        an array having the same shape as ``x`` and filled with zeros.
    """
    return _impl.zeros_like(x, dtype=dtype, device=device)
