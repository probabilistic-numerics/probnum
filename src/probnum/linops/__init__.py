"""Finite-dimensional Linear Operators."""

from ._kronecker import Kronecker, SymmetricKronecker, Symmetrize
from ._linear_operator import Identity, LinearOperator, Matrix
from ._scaling import Scaling
from ._utils import LinearOperatorLike, aslinop

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "aslinop",
    "LinearOperator",
    "Matrix",
    "Scaling",
    "Identity",
    "Kronecker",
    "SymmetricKronecker",
    "Symmetrize",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linops"

Matrix.__module__ = "probnum.linops"
Identity.__module__ = "probnum.linops"

Scaling.__module__ = "probnum.linops"

Kronecker.__module__ = "probnum.linops"
SymmetricKronecker.__module__ = "probnum.linops"
Symmetrize.__module__ = "probnum.linops"

aslinop.__module__ = "probnum.linops"
