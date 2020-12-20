"""Linear Algebra.

This package implements probabilistic numerical methods for the solution
of problems arising in linear algebra, such as the solution of linear
systems :math:`Ax=b`.
"""
from ._problinsolve import bayescg, problinsolve

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "problinsolve",
    "bayescg",
]
