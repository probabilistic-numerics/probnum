from typing import Optional

import numpy as np

import probnum
from probnum.linalg.solvers.stop_criteria._stopping_criterion import StoppingCriterion
from probnum.problems import LinearSystem
from probnum.type import ScalarArgType

__all__ = ["PosteriorContraction"]


class PosteriorContraction(StoppingCriterion):
    """Posterior contraction stopping criterion.

    Terminate when the uncertainty about the solution is sufficiently small, i.e. if it
    satisfies :math:`\\sqrt{\\operatorname{tr}(\\mathbb{Cov}(\\mathsf{x}))}
    \\leq \\max(\\text{atol}, \\text{rtol} \\lVert b \\rVert)`.

    Parameters
    ----------
    atol :
        Absolute residual tolerance.
    rtol :
        Relative residual tolerance.
    """

    def __init__(self, atol: ScalarArgType = 10 ** -5, rtol: ScalarArgType = 10 ** -5):
        self.atol = atol
        self.rtol = rtol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> bool:
        # Trace of the solution covariance
        trace_sol_cov = belief.x.cov.trace()

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(np.squeeze(belief.b.mean))
        return (
            np.abs(trace_sol_cov) <= self.atol ** 2
            or np.abs(trace_sol_cov) <= (self.rtol * b_norm) ** 2
        )
