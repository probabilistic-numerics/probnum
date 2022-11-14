"""Data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .. import BACKEND, Array, Backend
from ..typing import DTypeLike

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = [
    "DType",
    "bool",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "MachineLimitsFloatingPoint",
    "MachineLimitsInteger",
    "asdtype",
    "can_cast",
    "cast",
    "finfo",
    "iinfo",
    "is_floating_dtype",
    "promote_types",
    "result_type",
]

DType = _impl.DType
bool = _impl.bool
int32 = _impl.int32
int64 = _impl.int64
float16 = _impl.float16
float32 = _impl.float32
float64 = _impl.float64
complex64 = _impl.complex64
complex128 = _impl.complex128


@dataclass
class MachineLimitsFloatingPoint:
    """Machine limits for a floating point type.

    Parameters
    ----------
    bits
        The number of bits occupied by the type.
    max
        The largest representable number.
    min
        The smallest representable number, typically ``-max``.
    eps
        The difference between 1.0 and the next smallest representable float larger than 1.0. For example, for 64-bit binary floats in the IEEE-754 standard,
        ``eps = 2**-52``, approximately 2.22e-16.
    """

    bits: int
    eps: float
    max: float
    min: float


@dataclass
class MachineLimitsInteger:
    """Machine limits for an integer type.

    Parameters
    ----------
    bits
        The number of bits occupied by the type.
    max
        The largest representable number.
    min
        The smallest representable number, typically ``-max``.
    """

    bits: int
    max: int
    min: int


def asdtype(x: DTypeLike, /) -> DType:
    """Convert the input to a :class:`~probnum.backend.DType`.

    Parameters
    ----------
    x
        Object which can be converted to a :class:`~probnum.backend.DType`.
    """
    return _impl.asdtype(x)


def cast(
    x: Array, dtype: DType, /, *, casting: str = "unsafe", copy: bool = True
) -> Array:
    """Copies an array to a specified data type irrespective of type-promotion rules.

    Parameters
    ----------
    x
        Array to cast.
    dtype
        Desired data type.
    casting
        Controls what kind of data casting may occur.
    copy
        Specifies whether to copy an array when the specified ``dtype`` matches the data type of the input array ``x``. If ``True``, a newly allocated array will always be returned. If ``False`` and the specified ``dtype`` matches the data type of the input array, the input array will be returned; otherwise, a newly allocated will be returned.

    Returns
    -------
    out
        An array having the specified data type and the same shape as ``x``.
    """
    return _impl.cast(x, dtype, casting=casting, copy=copy)


def can_cast(from_: Union[DType, Array], to: DType, /) -> bool:
    """Determines if one data type can be cast to another data type according the type
    promotion rules.

    Parameters
    ----------
    from_
        Input data type or array from which to cast.
    to
        Desired data type.

    Returns
    -------
    out
        ``True`` if the cast can occur according to the type promotion rules; otherwise, ``False``.
    """
    return _impl.can_cast(from_, to)


def finfo(type: Union[DType, Array], /) -> MachineLimitsFloatingPoint:
    """Machine limits for floating-point data types.

    Parameters
    ----------
    type
        The kind of floating-point data-type about which to get information. If complex, the information is about its component data type.

    Returns
    -------
    out
        :class:`~probnum.backend.MachineLimitsFloatingPoint` object containing
        information on machine limits for floating-point data types.
    """
    return MachineLimitsFloatingPoint(**_impl.finfo(type))


def iinfo(type: Union[DType, Array], /) -> MachineLimitsInteger:
    """Machine limits for integer data types.

    Parameters
    ----------
    type
        The kind of integer data-type about which to get information.

    Returns
    -------
    out
        :class:`~probnum.backend.MachineLimitsInteger` object containing information on
        machine limits for integer data types.
    """
    return MachineLimitsInteger(**_impl.iinfo(type))


def is_floating_dtype(dtype: DType, /) -> bool:
    """Check whether ``dtype`` is a floating point data type.

    Parameters
    ----------
    dtype
        DType object to check.
    """
    return _impl.is_floating_dtype(dtype)


def promote_types(type1: DType, type2: DType, /) -> DType:
    """Returns the data type with the smallest size and smallest scalar kind to which
    both ``type1`` and ``type2`` may be safely cast.

    This function is symmetric, but rarely associative.

    Parameters
    ----------
    dtype1
        First data type.
    dtype2
        Second data type.

    Returns
    -------
    out
        The promoted data type.
    """
    return _impl.promote_types(type1, type2)


def result_type(*arrays_and_dtypes: Union[Array, DType]) -> DType:
    """Returns the dtype that results from applying the type promotion rules to the
    arguments.

    .. note::
       If provided mixed dtypes (e.g., integer and floating-point), the returned dtype will be implementation-specific.

    Parameters
    ----------
    arrays_and_dtypes
        An arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    out
        The dtype resulting from an operation involving the input arrays and dtypes.
    """
    return _impl.result_type(*arrays_and_dtypes)
