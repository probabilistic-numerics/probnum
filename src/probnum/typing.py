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
import scipy.sparse
from numpy.typing import ArrayLike  # pylint: disable=unused-import
from numpy.typing import DTypeLike as _NumPyDTypeLike

########################################################################################
# API Types
########################################################################################

ShapeType = Tuple[int, ...]

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
function :func:`probnum.utils.as_shape` before further internal processing."""

DTypeLike = _NumPyDTypeLike
"""Type of a public API argument for supplying an array's dtype.

Values of this type should always be converted into :class:`np.dtype`\\ s before further
internal processing."""

_ArrayIndexLike = Union[
    int,
    slice,
    type(Ellipsis),
    None,
    np.newaxis,
    np.ndarray,
]
ArrayIndicesLike = Union[_ArrayIndexLike, Tuple[_ArrayIndexLike, ...]]
"""Type of the argument to the :meth:`__getitem__` method of a NumPy-like array type
such as :class:`np.ndarray`, :class:`probnum.linops.LinearOperator` or
:class:`probnum.randvars.RandomVariable`."""

# Scalars, Arrays and Matrices
ScalarLike = Union[int, float, complex, numbers.Number, np.number]
"""Type of a public API argument for supplying a scalar value.

Values of this type should always be converted into :class:`np.number`\\ s using the
function :func:`probnum.utils.as_scalar` before further internal processing."""

LinearOperatorArgType = Union[
    np.ndarray,
    scipy.sparse.spmatrix,
    "probnum.linops.LinearOperator",
]
"""Type of a public API argument for supplying a matrix or finite-dimensional linear operator."""

########################################################################################
# Other Types
########################################################################################

ToleranceDiffusionType = Union[FloatLike, np.ndarray]
r"""Type of a quantity that describes tolerances, errors, and diffusions.

Used for absolute (atol) and relative tolerances (rtol), local error estimates, as well as
(the diagonal entries of diagonal matrices representing) diffusion models.
atol, rtol, and diffusion are usually floats, but can be generalized to arrays -- essentially,
to every :math:`\tau` that allows arithmetic operations such as

.. math:: \tau + tau * \text{vec}, \text{ or } L \otimes \text{diag}(\tau)

respectively. Currently, the array-support for diffusions is experimental (at best).
"""

DenseOutputLocationArgType = Union[FloatLike, np.ndarray]
"""TimeSeriesPosteriors and derived classes can be evaluated at a single location 't'
or an array of locations."""

NotImplementedType = type(NotImplemented)
