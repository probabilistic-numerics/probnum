"""Type aliases for the backend."""

from __future__ import annotations

import numbers
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike as _NumPyArrayLike, DTypeLike as _NumPyDTypeLike

from ._array_object import Array, Scalar

__all__ = [
    # API Types
    "ShapeType",
    "SeedType",
    # Argument Types
    "IntLike",
    "FloatLike",
    "ShapeLike",
    "DTypeLike",
    "ArrayIndicesLike",
    "ScalarLike",
    "ArrayLike",
    "SeedLike",
    "NotImplementedType",
]

########################################################################################
# API Types
########################################################################################

# Array Utilities
ShapeType = Tuple[int, ...]
"""Type defining a shape of an object."""

# Random Number Generation
SeedType = "probnum.backend.random._SeedType"
"""Type defining the seed of a random number generator."""

########################################################################################
# Argument Types
########################################################################################

# Python Numbers
IntLike = Union[int, numbers.Integral, np.integer]
"""Object that can be converted to an integer.

Arguments of type :attr:`IntLike` should always be converted into :class:`int`\\ s
before further internal processing."""

FloatLike = Union[float, numbers.Real, np.floating]
"""Object that can be converted to a float.

Arguments of type :attr:`FloatLike` should always be converteg into :class:`float`\\ s
before further internal processing."""

# Scalars, Arrays and Matrices
ScalarLike = Union[Scalar, int, float, complex, numbers.Number, np.number]
"""Object that can be converted to a scalar value.

Arguments of type :attr:`ScalarLike` should always be converted into objects of
:attr:ScalarType` using the function :func:`backend.as_scalar` before further internal
processing."""

ArrayLike = Union[Array, _NumPyArrayLike]
"""Object that can be converted to an array.

Arguments of type :attr:`ArrayLike` should always be converted into objects of
:attr:`ArrayType`\\ s using the function :func:`backend.asarray` before further internal
processing."""

# Array Utilities
ShapeLike = Union[IntLike, Iterable[IntLike]]
"""Object that can be converted to a shape.

Arguments of type :attr:`ShapeLike` should always be converted into :class:`ShapeType` using the
function :func:`backend.as_shape` before further internal processing."""

DTypeLike = Union["probnum.backend.dtype", _NumPyDTypeLike]
"""Object that can be converted to an array dtype.

Arguments of type :attr:`DTypeLike` should always be converted into :class:`backend.dtype`\\ s before further
internal processing."""

_ArrayIndexLike = Union[
    int,
    slice,
    type(Ellipsis),
    None,
    "probnum.backend.newaxis",
    ArrayLike,
]
ArrayIndicesLike = Union[_ArrayIndexLike, Tuple[_ArrayIndexLike, ...]]
"""Object that can be converted to indices of an array.

Type of the argument to the :meth:`__getitem__` method of an :class:`Array` or similar
object.
"""

# Random Number Generation
SeedLike = Optional[int]
"""Type of a public API argument for supplying the seed of a random number generator.

Values of this type should always be converted to :class:`SeedType` using the function
:func:`backend.random.seed` before further internal processing."""


########################################################################################
# Other Types
########################################################################################

NotImplementedType = type(NotImplemented)
"""Type of the `NotImplemented` constant."""
