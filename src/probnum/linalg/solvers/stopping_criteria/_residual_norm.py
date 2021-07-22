"""Stopping criterion based on the norm of the residual."""

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum.typing import ScalarArgType

from ._linear_solver_stopping_criterion import LinearSolverStoppingCriterion


class ResidualNormStopCrit(LinearSolverStoppingCriterion):
    r"""Residual stopping criterion.

    Terminate when the euclidean norm of the residual :math:`r_{i} = A x_{i} - b` is
    sufficiently small, i.e. if it satisfies :math:`\lVert r_i \rVert_2 \leq \max(
    \text{atol}, \text{rtol} \lVert b \rVert_2)`.

    Parameters
    ----------
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    """

    def __init__(
        self,
        atol: ScalarArgType = 10 ** -5,
        rtol: ScalarArgType = 10 ** -5,
    ):
        self.atol = atol
        self.rtol = rtol

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> bool:

        residual_norm = np.linalg.norm(solver_state.residual.flatten(), ord=2)
        b_norm = np.linalg.norm(
            solver_state.problem.b.flatten(), ord=2
        )  # TODO: cache this
        return residual_norm <= self.atol or residual_norm <= self.rtol * b_norm
