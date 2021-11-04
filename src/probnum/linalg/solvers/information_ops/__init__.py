r"""Information operators of probabilistic linear solvers.

Information operators collect information about the quantity of interest
by observing the numerical problem to be solved given an action. When solving
linear systems, the information operator takes an action vector and observes
the tuple :math:`(A, b)`, returning an observation vector. For example, one might
observe the right hand side :math:`y = b^\top s = (Ax)^\top s` with the action :math:`s`.
"""

from ._linear_solver_information_op import LinearSolverInformationOp
from ._matvec import MatVecInformationOp
from ._projected_rhs import ProjectedRHSInformationOp

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverInformationOp",
    "MatVecInformationOp",
    "ProjectedRHSInformationOp",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverInformationOp.__module__ = "probnum.linalg.solvers.information_ops"
MatVecInformationOp.__module__ = "probnum.linalg.solvers.information_ops"
ProjectedRHSInformationOp.__module__ = "probnum.linalg.solvers.information_ops"
