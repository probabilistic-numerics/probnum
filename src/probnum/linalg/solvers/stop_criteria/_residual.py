from typing import Optional

import numpy as np

from probnum.linalg.solvers._state import LinearSolverState
from probnum.linalg.solvers.stop_criteria._stopping_criterion import StoppingCriterion
from probnum.problems import LinearSystem
from probnum.type import ScalarArgType

__all__ = ["ResidualNorm"]


class ResidualNorm(StoppingCriterion):
    """Residual stopping criterion.

    Terminate when the norm of the residual :math:`r_{i} = A x_{i} - b` is
    sufficiently small, i.e. if it satisfies :math:`\\lVert r_i \\rVert \\leq \\max(
    \\text{atol}, \\text{rtol} \\lVert b \\rVert)`.

    Parameters
    ----------
    atol :
        Absolute residual tolerance.
    rtol :
        Relative residual tolerance.
    norm_ord :
        Order of the norm. Defaults to the euclidean (:math:`p=2`) norm. See
        :func:`numpy.linalg.norm` for a complete list of available choices.
    """

    def __init__(
        self, atol: ScalarArgType = 10 ** -5, rtol: ScalarArgType = 10 ** -5, norm_ord=2
    ):
        self.atol = atol
        self.rtol = rtol
        self.norm_ord = norm_ord
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> bool:
        if solver_state is None:
            solver_state = LinearSolverState(problem=problem, belief=belief)

        # Compute residual norm
        residual_norm = np.linalg.norm(
            solver_state.cache.residual.flatten(), ord=self.norm_ord
        )

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b.flatten(), ord=self.norm_ord)
        return residual_norm <= self.atol or residual_norm <= self.rtol * b_norm
