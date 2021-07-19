"""Linear Algebra.

This package implements probabilistic numerical methods for the solution
of problems arising in linear algebra, such as the solution of linear
systems :math:`Ax=b`.
"""
from probnum.linalg._problinsolve import bayescg, problinsolve

from . import solvers

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "problinsolve",
    "bayescg",
]

# Set correct module paths. Corrects links and module paths in documentation.
problinsolve.__module__ = "probnum.linalg"
bayescg.__module__ = "probnum.linalg"
