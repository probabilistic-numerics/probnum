"""Information operator returning a projection of the residual."""
import numpy as np

import probnum  # pylint: disable="unused-import"

from ._linear_solver_info_op import LinearSolverInfoOp


class ProjResidualInfoOp(LinearSolverInfoOp):
    r"""Projected residual :math:`s_i \mapsto s_i^\top (A x_i-b)` of the linear system.

    Obtain information about a linear system by projecting the current
    residual :math:`r_i = A x_i - b` onto a given action :math:`s_i` resulting
    in :math:`y_i = s_i^\top r_i`.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> np.ndarray:
        r"""Projected residual :math:`s_i^\top (A x_i - b)` of the linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        return solver_state.action @ solver_state.residual
