"""Stopping criterion based on the uncertainty about a quantity of interest."""

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum.typing import ScalarArgType

from ._linear_solver_stopping_criterion import LinearSolverStoppingCriterion


class PosteriorContractionStopCrit(LinearSolverStoppingCriterion):
    r"""Posterior contraction stopping criterion.

    Terminate when the uncertainty about the solution is sufficiently small, i.e. if it
    satisfies :math:`\sqrt{\operatorname{tr}(\mathbb{Cov}(\mathsf{x}))}
    \leq \max(\text{atol}, \text{rtol} \lVert b \rVert_2)`.

    Parameters
    ----------
    atol :
        Absolute residual tolerance.
    rtol :
        Relative residual tolerance.
    """

    def __init__(
        self,
        atol: ScalarArgType = 10 ** -5,
        rtol: ScalarArgType = 10 ** -5,
    ):
        self.atol = probnum.utils.as_numpy_scalar(atol)
        self.rtol = probnum.utils.as_numpy_scalar(rtol)

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> bool:

        trace_sol_cov = solver_state.belief.x.cov.trace()
        b_norm = np.linalg.norm(solver_state.problem.b, ord=2)

        return (
            np.abs(trace_sol_cov) <= self.atol ** 2
            or np.abs(trace_sol_cov) <= (self.rtol * b_norm) ** 2
        )
