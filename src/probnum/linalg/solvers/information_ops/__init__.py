r"""Information operators of probabilistic linear solvers.

Information operators collect information about the quantity of interest
by observing the numerical problem to be solved given an action. When solving
linear systems, the information operator takes an action vector and observes
the tuple :math:`(A, b)`, returning an observation vector. For example, one might
observe the projected residual :math:`y = s^\top (A x_i - b)` with the action :math:`s`.
"""

from ._linear_solver_info_op import LinearSolverInfoOp
from ._matvec import MatVecInfoOp
from ._projected_residual import ProjectedResidualInfoOp

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverInfoOp",
    "MatVecInfoOp",
    "ProjectedResidualInfoOp",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverInfoOp.__module__ = "probnum.linalg.solvers.information_ops"
MatVecInfoOp.__module__ = "probnum.linalg.solvers.information_ops"
ProjectedResidualInfoOp.__module__ = "probnum.linalg.solvers.information_ops"
