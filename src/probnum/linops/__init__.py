"""Finite-dimensional Linear Operators."""

from ._diagonal import Diagonal, Identity
from ._kronecker import Kronecker, SymmetricKronecker, Symmetrize
from ._linear_operator import LinearOperator, Matrix
from ._utils import aslinop

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "aslinop",
    "LinearOperator",
    "Matrix",
    "Diagonal",
    "Identity",
    "Kronecker",
    "SymmetricKronecker",
    "Symmetrize",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linops"

Matrix.__module__ = "probnum.linops"

Diagonal.__module__ = "probnum.linops"
Identity.__module__ = "probnum.linops"

Kronecker.__module__ = "probnum.linops"
SymmetricKronecker.__module__ = "probnum.linops"
Symmetrize.__module__ = "probnum.linops"

aslinop.__module__ = "probnum.linops"
