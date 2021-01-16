from typing import Optional

import probnum
from probnum.linalg.linearsolvers.stop_criteria._stopping_criterion import (
    StoppingCriterion,
)
from probnum.problems import LinearSystem
from probnum.type import IntArgType

# Public classes and functions. Order is reflected in documentation.
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
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> bool:
        if self.maxiter is None:
            _maxiter = problem.A.shape[0] * 10
        else:
            _maxiter = self.maxiter

        try:
            return solver_state.iteration >= _maxiter
        except AttributeError:
            return False
