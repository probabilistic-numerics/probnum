from typing import Optional

import numpy as np

import probnum
from probnum.linalg.solvers._state import LinearSolverState
from probnum.linalg.solvers.data import LinearSolverAction
from probnum.linalg.solvers.policies._policy import Policy
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["MaxSupNormColumn"]


class MaxSupNormColumn(Policy):
    r"""Policy returning a unit vector selecting the column of the system matrix with
    maximum supremum norm.

    Returns a unit action :math:`e_i` which, when applied to the system matrix selects
    the column with the largest supremum norm :math:`\lVert A e_i \rVert_{\infty} \geq
    \lVert A e_j\rVert_{\infty}`, i.e. the column with the largest absolute value.
    """

    def __init__(self):
        super().__init__(policy=self.__call__, is_deterministic=True)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> LinearSolverAction:

        if solver_state is None:
            solver_state = LinearSolverState(problem=problem, belief=belief)

        # Column supremum norms
        col_norms = np.linalg.norm(solver_state.problem.A, ord=np.inf, axis=1)

        # Maximum norm column
        maxnorm_idx = np.argmax(col_norms)

        # Unit vector
        unitvec = np.eye(N=solver_state.problem.A.shape[1], M=1, k=-maxnorm_idx)

        return LinearSolverAction(actA=unitvec)
