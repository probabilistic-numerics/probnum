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
internal representation of those same objects. As an example, a user might pass an
object which can be converted to a finite dimensional linear operator. This argument
could be an class:`~probnum.backend.Array`, a sparse matrix
:class:`~scipy.sparse.spmatrix` or a :class:`~probnum.linops.LinearOperator`. The type
alias :attr:`LinearOperatorLike`combines all these in a single type. Internally, the
passed argument is then converted to a :class:`~probnum.linops.LinearOperator`.
"""

from __future__ import annotations

from typing import Union

import scipy.sparse

from probnum import backend
from probnum.backend.typing import ArrayLike

__all__ = [
    # API Types
    "MatrixType",
    # Argument Types
    "LinearOperatorLike",
]

########################################################################################
# API Types
########################################################################################

# Scalars, Arrays and Matrices
MatrixType = Union[backend.Array, "probnum.linops.LinearOperator"]
"""Type defining a matrix, i.e. a linear map between finite-dimensional vector spaces.

An object :code:`matrix`, which behaves like an :class:`~probnum.backend.Array` and
satisfies :code:`matrix.ndim == 2`.
"""

########################################################################################
# Argument Types
########################################################################################

# Scalars, Arrays and Matrices
LinearOperatorLike = Union[
    ArrayLike,
    scipy.sparse.spmatrix,
    "probnum.linops.LinearOperator",
]
"""Object that can be converted to a :class:`~probnum.linops.LinearOperator`.

Arguments of type :attr:`LinearOperatorLike` should always be converted into
:class:`~probnum.linops.LinearOperator`\\ s using the function
:func:`probnum.linops.aslinop` before further internal processing."""
