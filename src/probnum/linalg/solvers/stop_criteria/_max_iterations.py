from typing import Optional

import probnum
from probnum.linalg.solvers._state import LinearSolverState
from probnum.linalg.solvers.stop_criteria._stopping_criterion import StoppingCriterion
from probnum.problems import LinearSystem
from probnum.type import IntArgType

__all__ = ["MaxIterations"]


class MaxIterations(StoppingCriterion):
    """Maximum number of iterations.

    Stop when a maximum number of iterations is reached. If none is
    specified, defaults to :math:`10n`, where :math:`n` is the dimension
    of the solution to the linear system.
    """

    def __init__(self, maxiter: IntArgType = None):
        self.maxiter = maxiter
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> bool:
        if self.maxiter is None:
            _maxiter = problem.A.shape[0] * 10
        else:
            _maxiter = self.maxiter

        if solver_state is None:
            solver_state = LinearSolverState(problem=problem, belief=belief)

        return solver_state.info.iteration >= _maxiter
