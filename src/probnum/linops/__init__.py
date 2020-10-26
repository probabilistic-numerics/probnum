"""
Finite-dimensional Linear Operators.

This module defines classes and methods that implement finite dimensional linear
operators. It can be used to do linear algebra with (structured) matrices without
explicitly representing them in memory. This often allows for the definition of a more
efficient matrix-vector product. Linear operators can be applied, added, multiplied,
transposed, and more as one would expect from matrix algebra.

Several algorithms in the :mod:`probnum.linalg` library are able to operate on
:class:`LinearOperator` instances.
"""

from ._kronecker import Kronecker, Svec, SymmetricKronecker, Symmetrize, Vec
from ._linear_operator import Identity, LinearOperator, MatrixMult, ScalarMult
from ._utils import aslinop

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "aslinop",
    "LinearOperator",
    "Identity",
    "ScalarMult",
    "MatrixMult",
    "Kronecker",
    "SymmetricKronecker",
    "Symmetrize",
    "Vec",
    "Svec",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linops"

Identity.__module__ = "probnum.linops"
ScalarMult.__module__ = "probnum.linops"
MatrixMult.__module__ = "probnum.linops"

Kronecker.__module__ = "probnum.linops"
Svec.__module__ = "probnum.linops"
SymmetricKronecker.__module__ = "probnum.linops"
Symmetrize.__module__ = "probnum.linops"
Vec.__module__ = "probnum.linops"

aslinop.__module__ = "probnum.linops"
