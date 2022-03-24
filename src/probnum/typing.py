"""Custom type aliases.

This module defines commonly used types in the library. These are separated into two
different kinds, API types and argument types.

**API types** (``*Type``) are aliases which define custom types used throughout the
library. Objects of this type may be supplied as arguments or returned by a method.

**Argument types** (``*Like``) are aliases which define commonly used method
arguments that are internally converted to a standardized representation. These should
only ever be used in the signature of a method and then be converted internally, e.g. in
a class instantiation or an interface. They enable the user to conveniently supply a
variety of objects of different types for the same argument, while ensuring a unified
internal representation of those same objects. As an example, take the different ways a
user might specify a shape: ``2``, ``(2,)``, ``[2, 2]``. These may all be acceptable
arguments to a function taking a shape, but internally should always be converted to a
:attr:`ShapeType`, i.e. a tuple of ``int``\\ s.
"""

from __future__ import annotations

import numbers
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike as _NumPyArrayLike, DTypeLike as _NumPyDTypeLike
import scipy.sparse

__all__ = [
    # API Types
    "ScalarType",
    "ArrayType",
    "MatrixType",
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
    "LinearOperatorLike",
    "SeedLike",
    "NotImplementedType",
]

########################################################################################
# API Types
########################################################################################

# Scalars, Arrays and Matrices
ScalarType = "probnum.backend._ArrayType"
"""Type defining a scalar."""

ArrayType = "probnum.backend._ArrayType"
"""Type defining a (possibly multi-dimensional) array."""

MatrixType = Union[ArrayType, "probnum.linops.LinearOperator"]
"""Type defining a matrix, i.e. a linear map between finite-dimensional vector spaces.

An object :code:`matrix` of :attr:`MatrixType`, which behaves like an object of
:class:`ArrayType` with :code:`matrix.ndim == 2`.
"""

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
ScalarLike = Union[ScalarType, int, float, complex, numbers.Number, np.number]
"""Object that can be converted to a scalar value.

Arguments of type :attr:`ScalarLike` should always be converted into objects of
:attr:ScalarType` using the function :func:`backend.as_scalar` before further internal
processing."""

ArrayLike = Union[ArrayType, _NumPyArrayLike]
"""Object that can be converted to an array.

Arguments of type :attr:`ArrayLike` should always be converted into objects of
:attr:`ArrayType`\\ s using the function :func:`backend.asarray` before further internal
processing."""

LinearOperatorLike = Union[
    ArrayLike,
    scipy.sparse.spmatrix,
    "probnum.linops.LinearOperator",
]
"""Object that can be converted to a :class:`~probnum.linops.LinearOperator`.

Arguments of type :attr:`LinearOperatorLike` should always be converted into
:class:`~probnum.linops.LinearOperator`\\ s using the function
:func:`probnum.linops.aslinop` before further internal processing."""

# Array Utilities
ShapeLike = Union[IntLike, Iterable[IntLike]]
"""Object that can be converted to a shape.

Arguments of type :attr:`ShapeLike` should always be converted into :class:`ShapeType` using the
function :func:`probnum.utils.as_shape` before further internal processing."""

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

Type of the argument to the :meth:`__getitem__` method of a NumPy-like array type
such as :class:`numpy.ndarray`, :class:`probnum.linops.LinearOperator` or
:class:`probnum.randvars.RandomVariable`."""

# Random Number Generation
SeedLike = Optional[int]
"""Type of a public API argument for supplying the seed of a random number generator.

Values of this type should always be converted to :class:`SeedType` using the function
:func:`probnum.backend.random.seed` before further internal processing."""

########################################################################################
# Other Types
########################################################################################

NotImplementedType = type(NotImplemented)
"""Type of the `NotImplemented` constant."""
