"""Information operator returning a projection of the right-hand-side."""
import numpy as np

import probnum  # pylint: disable="unused-import"

from ._linear_solver_information_op import LinearSolverInformationOp


class ProjectedResidualInformationOp(LinearSolverInformationOp):
    r"""Projected residual information operator.

    Obtain information about a linear system by projecting the residual
    :math:`b-Ax_{i-1}` onto a given action :math:`s_i` resulting in :math:`s_i
    \mapsto s_i^\top r_{i-1} = s_i^\top (b - A x_{i-1}) = s_i^\top
    A (x - x_{i-1})`.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> np.ndarray:
        r"""Projected residual :math:`s_i^\top r_{i-1} = s_i^\top (b - A x_{i-1})`
        of the linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        return solver_state.action.T @ solver_state.residual
