from typing import Optional

import numpy as np

import probnum
from probnum.linalg.linearsolvers.stop_criteria._stopping_criterion import (
    StoppingCriterion,
)
from probnum.problems import LinearSystem
from probnum.type import ScalarArgType

# Public classes and functions. Order is reflected in documentation.
__all__ = ["Residual"]


class Residual(StoppingCriterion):
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
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> bool:
        # Compute residual norm
        try:
            residual = solver_state.residual
        except AttributeError:
            residual = problem.A @ belief.x.mean - problem.b
        residual_norm = np.linalg.norm(residual.flatten(), ord=self.norm_ord)

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b.flatten(), ord=self.norm_ord)
        return residual_norm <= self.atol or residual_norm <= self.rtol * b_norm
