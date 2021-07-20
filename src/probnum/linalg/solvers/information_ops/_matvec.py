"""Information operator returning a matrix-vector product with the matrix."""
import numpy as np

import probnum  # pylint: disable="unused-import"

from ._linear_solver_info_op import LinearSolverInfoOp


class MatVecInfoOp(LinearSolverInfoOp):
    r"""Matrix-vector product :math:`s_i \mapsto As_i` with the system matrix.

    Obtain information about a linear system by multiplying an action :math:`s_i`
    with the system matrix giving :math:`y_i = A s_i`.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> np.ndarray:
        return solver_state.problem.A @ solver_state.action
