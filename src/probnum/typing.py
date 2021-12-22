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

from __future__ import annotations

import numbers
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from numpy.typing import ArrayLike as _NumPyArrayLike, DTypeLike as _NumPyDTypeLike

########################################################################################
# API Types
########################################################################################

# Array Utilities
ShapeType = Tuple[int, ...]

# Scalars, Arrays and Matrices
ScalarType = "probnum.backend.ndarray"
MatrixType = Union["probnum.backend.ndarray", "probnum.linops.LinearOperator"]

# Random Number Generation
SeedType = Union[np.random.SeedSequence, "jax.random.PRNGKey"]


########################################################################################
# Argument Types
########################################################################################

# Python Numbers
IntLike = Union[int, numbers.Integral, np.integer]
"""Type of a public API argument for supplying an integer.

Values of this type should always be converted into :class:`int`\\ s before further
internal processing."""

FloatLike = Union[float, numbers.Real, np.floating]
"""Type of a public API argument for supplying a float.

Values of this type should always be converteg into :class:`float`\\ s before further
internal processing."""

# Array Utilities
ShapeLike = Union[IntLike, Iterable[IntLike]]
"""Type of a public API argument for supplying a shape.

Values of this type should always be converted into :class:`ShapeType` using the
function :func:`probnum.backend.as_shape` before further internal processing."""

DTypeLike = Union[_NumPyDTypeLike, "jax.numpy.dtype", "torch.dtype"]
"""Type of a public API argument for supplying an array's dtype.

Values of this type should always be converted into :class:`backend.dtype`\\ s using the
function :func:`probnum.backend.as_dtype` before further internal processing."""

_ArrayIndexLike = Union[
    int,
    slice,
    type(Ellipsis),
    None,
    "probnum.backend.newaxis",
    "probnum.backend.ndarray",
]
ArrayIndicesLike = Union[_ArrayIndexLike, Tuple[_ArrayIndexLike, ...]]
"""Type of the argument to the :meth:`__getitem__` method of a NumPy-like array type
such as :class:`probnum.backend.ndarray`, :class:`probnum.linops.LinearOperator` or
:class:`probnum.randvars.RandomVariable`."""

# Scalars, Arrays and Matrices
ScalarLike = Union[ScalarType, int, float, complex, numbers.Number, np.number]
"""Type of a public API argument for supplying a scalar value.

Values of this type should always be converted into :class:`ScalarType`\\ s using
the function :func:`probnum.backend.as_scalar` before further internal processing."""

ArrayLike = Union[_NumPyArrayLike, "jax.numpy.ndarray", "torch.Tensor"]
"""Type of a public API argument for supplying an array.

Values of this type should always be converted into :class:`backend.ndarray`\\ s using
the function :func:`probnum.backend.as_array` before further internal processing."""

LinearOperatorLike = Union[
    ArrayLike,
    scipy.sparse.spmatrix,
    "probnum.linops.LinearOperator",
]
"""Type of a public API argument for supplying a finite-dimensional linear operator.

Values of this type should always be converted into :class:`probnum.linops.\\
LinearOperator`\\ s using the function :func:`probnum.linops.as_linop` before further
internal processing."""

# Random Number Generation
SeedLike = Optional[int]
"""Type of a public API argument for supplying the seed of a random number generator.

Values of this type should always be converted to :class:`SeedType` using the function
:func:`probnum.backend.random.seed` before further internal processing."""

########################################################################################
# Other Types
########################################################################################

NotImplementedType = type(NotImplemented)
