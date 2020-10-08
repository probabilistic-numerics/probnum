"""
(Finite-dimensional) Linear Operators.

This package implements a variety of finite dimensional linear operators. These have the
advantage of only implementing a matrix-vector product instead of representing the full
linear operator as a matrix in memory.
"""

from ._kronecker import *
from ._linear_operator import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearOperator",
    "Identity",
    "ScalarMult",
    "MatrixMult",
    "Kronecker",
    "SymmetricKronecker",
    "Symmetrize",
    "aslinop",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linops"
ScalarMult.__module__ = "probnum.linops"
