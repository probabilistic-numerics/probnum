"""Stopping criterion based on a maximum number of iterations."""
from typing import Optional

import probnum  # pylint: disable="unused-import"

from ._linear_solver_stopping_criterion import LinearSolverStoppingCriterion


class MaxIterationsStoppingCriterion(LinearSolverStoppingCriterion):
    r"""Stop after a maximum number of iterations.

    Stop when the solver has taken a maximum number of steps. If ``None`` is
    specified, defaults to :math:`10n`, where :math:`n` is the dimension
    of the solution to the linear system.

    Parameters
    ----------
    maxiter
        Maximum number of steps the solver should take.
    """

    def __init__(self, maxiter: Optional[int] = None):
        self.maxiter = maxiter

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> bool:
        """Check whether the maximum number of iterations has been reached.

        Parameters
        ----------
        solver_state
            Current state of the linear solver.
        """
        if self.maxiter is None:
            return solver_state.step >= solver_state.problem.A.shape[0] * 10

        return solver_state.step >= self.maxiter
