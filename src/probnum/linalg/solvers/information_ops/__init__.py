"""Information operators of probabilistic linear solvers."""

from ._linear_solver_info_op import LinearSolverInfoOp
from ._matvec import MatVecInfoOp
from ._proj_residual import ProjResidualInfoOp

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverInfoOp",
    "MatVecInfoOp",
    "ProjResidualInfoOp",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverInfoOp.__module__ = "probnum.linalg.solvers.information_ops"
MatVecInfoOp.__module__ = "probnum.linalg.solvers.information_ops"
ProjResidualInfoOp.__module__ = "probnum.linalg.solvers.information_ops"
