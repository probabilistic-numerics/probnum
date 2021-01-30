"""Finite-dimensional Linear Operators.

This package implements finite dimensional linear operators. It can be used to
represent linear maps between finite-dimensional vector spaces without explicitly
constructing their matrix representation in memory. This is particularly useful for
sparse and structured matrices and often allows for the definition of a more
efficient matrix-vector product. Linear operators support common algebraic
operations, including matrix-vector products, addition, multiplication, and
transposition.

Several algorithms in the :mod:`probnum.linalg` subpackage are able to operate on
:class:`~probnum.linops.LinearOperator` instances.
"""

from ._kronecker import BoxProduct, Kronecker, SymmetricKronecker, Symmetrize
from ._linear_operator import DiagMult, Identity, LinearOperator, MatrixMult, ScalarMult
from ._projections import OrthogonalProjection
from ._utils import aslinop

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "aslinop",
    "LinearOperator",
    "Identity",
    "ScalarMult",
    "DiagMult",
    "MatrixMult",
    "OrthogonalProjection",
    "Kronecker",
    "BoxProduct",
    "SymmetricKronecker",
    "Symmetrize",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linops"

Identity.__module__ = "probnum.linops"
ScalarMult.__module__ = "probnum.linops"
DiagMult.__module__ = "probnum.linops"
MatrixMult.__module__ = "probnum.linops"

OrthogonalProjection.__module__ = "probnum.linops"

Kronecker.__module__ = "probnum.linops"
BoxProduct.__module__ = "probnum.linops"
SymmetricKronecker.__module__ = "probnum.linops"
Symmetrize.__module__ = "probnum.linops"

aslinop.__module__ = "probnum.linops"
