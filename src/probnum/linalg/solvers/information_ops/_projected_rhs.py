"""Information operator returning a projection of the residual."""
import numpy as np

import probnum  # pylint: disable="unused-import"

from ._linear_solver_information_op import LinearSolverInformationOp


class ProjectedRHSInformationOp(LinearSolverInformationOp):
    r"""Projected right hand side :math:`s_i \mapsto b^\top s_i = (Ax)^\top s_i` of the linear system.

    Obtain information about a linear system by projecting the right hand side :math:`b=Ax` onto a given action :math:`s_i` resulting in :math:`y_i = s_i^\top b`.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> np.ndarray:
        r"""Projected right hand side :math:`s_i^\top b = s_i^\top Ax` of the linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        return solver_state.action @ solver_state.problem.b
