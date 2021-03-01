"""Custom type aliases.

This module defines commonly used types in the library. These are separated into two
different kinds, API types and argument types.

API types are aliases which define custom types used throughout the library. Objects of
this type may be supplied as arguments or returned by a method.

Argument types are aliases which define commonly used method arguments. These should
only ever be used in the signature of a method and then be converted internally, e.g.
in a class instantiation or an interface. They enable the user to conveniently
specify a variety of object types for the same argument, while ensuring a unified
internal representation of those same objects.
"""

import numbers
from typing import Iterable, Tuple, Union

import numpy as np

########################################################################################
# API Types
########################################################################################

ShapeType = Tuple[int, ...]

RandomStateType = Union[np.random.RandomState, np.random.Generator]
"""Type of a random number generator."""

########################################################################################
# Argument Types
########################################################################################

IntArgType = Union[int, numbers.Integral, np.integer]
FloatArgType = Union[float, numbers.Real, np.floating]

ShapeArgType = Union[IntArgType, Iterable[IntArgType]]
"""Type of a public API argument for supplying a shape. Values of this type should
always be converted into :class:`ShapeType` using the function
:func:`probnum.utils.as_shape` before further internal processing."""

DTypeArgType = Union[np.dtype, str]
"""Type of a public API argument for supplying a dtype. Values of this type should
always be converted into :class:`np.dtype` using the function
:func:`np.dtype` before further internal processing."""

ScalarArgType = Union[int, float, complex, numbers.Number, np.number]
"""Type of a public API argument for supplying a scalar value. Values of this type
should always be converted into :class:`np.generic` using the function
:func:`probnum.utils.as_scalar` before further internal processing."""

ArrayLikeGetitemArgType = Union[
    int,
    slice,
    np.ndarray,
    np.newaxis,
    None,
    type(Ellipsis),
    Tuple[Union[int, slice, np.ndarray, np.newaxis, None, type(Ellipsis)], ...],
]

RandomStateArgType = Union[None, int, np.random.RandomState, np.random.Generator]
"""Type of a public API argument for supplying a random number generator. Values of this
type should always be converted into :class:`RandomStateType` using the function
:func:`probnum.utils.as_random_state` before further internal processing."""
