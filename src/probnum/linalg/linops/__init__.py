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

from probnum.linalg.linops.kronecker import *
from probnum.linalg.linops.linearoperators import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearOperator",
    "Identity",
    "ScalarMult",
    "MatrixMult",
    "Kronecker",
    "SymmetricKronecker",
    "Vec",
    "Svec",
    "Symmetrize",
    "aslinop",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linalg.linops"
ScalarMult.__module__ = "probnum.linalg.linops"
