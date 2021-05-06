from typing import Optional

import probnum  # pylint: disable="unused-import"
from probnum.linalg.solvers._state import LinearSolverState
from probnum.linalg.solvers.data import LinearSolverAction
from probnum.linalg.solvers.policies._policy import Policy
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["Residual"]


class Residual(Policy):
    r"""Policy returning the negative residual of the linear system.

    Returns an action given by the residual :math:`s_k = r_{k-1} = A x_{k-1} - b`.
    This can also be interpreted as the gradient of
    :math:`f(x)=\frac{1}{2}x^\top A x - b^\top x + c` assuming symmetric positive
    definite :math:`A`.
    """

    def __init__(self):
        super().__init__(policy=self.__call__, is_deterministic=True)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional[LinearSolverState] = None,
    ) -> LinearSolverAction:

        if solver_state is None:
            solver_state = LinearSolverState(problem=problem, belief=belief)

        action = solver_state.cache.residual

        return LinearSolverAction(actA=action)
