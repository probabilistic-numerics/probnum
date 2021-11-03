"""Matrix-based belief updates for the quantities of interest of a linear system."""

from ._matrix_based_linear_belief_update import MatrixBasedLinearBeliefUpdate
from ._symmetric_matrix_based_linear_belief_update import (
    SymmetricMatrixBasedLinearBeliefUpdate,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "MatrixBasedLinearBeliefUpdate",
    "SymmetricMatrixBasedLinearBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
MatrixBasedLinearBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates.matrix_based"
)

SymmetricMatrixBasedLinearBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates.matrix_based"
)
