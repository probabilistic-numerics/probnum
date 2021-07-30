"""Test problems from linear algebra."""

from ._random_linear_system import random_linear_system
from ._random_spd_matrix import random_sparse_spd_matrix, random_spd_matrix
from ._suitesparse_matrix import SuiteSparseMatrix, suitesparse_matrix

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "random_spd_matrix",
    "random_sparse_spd_matrix",
    "suitesparse_matrix",
    "random_linear_system",
    "SuiteSparseMatrix",
]

# Set correct module paths. Corrects links and module paths in documentation.
SuiteSparseMatrix.__module__ = "probnum.problems.zoo.linalg"
