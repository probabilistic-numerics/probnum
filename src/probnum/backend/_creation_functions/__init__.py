"""Array creation functions."""

from __future__ import annotations

from typing import Optional, Union

from .. import BACKEND, Array, Backend, Dtype, Scalar, ndim
from ..typing import DTypeLike, ScalarLike

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["asscalar", "asarray", "tril", "triu"]


def asarray(
    obj: Union[Array, bool, int, float, "NestedSequence", "SupportsBufferProtocol"],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional["probnum.backend.Device"] = None,
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
