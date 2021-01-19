from typing import Optional, Tuple

import numpy as np

import probnum
from probnum.linalg.solvers.policies._policy import Policy, _get_residual
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ConjugateDirections"]


class ConjugateDirections(Policy):
    """Policy returning :math:`A`-conjugate directions.

    Returns an action given by :math:`s_i = -\\mathbb{E}[\\mathsf{H}]r_{i-1}` where
    :math:`r_{i-1} = A x_{i-1} - b` is the current residual. If the posterior mean of
    :math:`\\mathbb{E}[\\mathsf{H}]` of the inverse model equals the true inverse,
    the resulting action is the exact step to the solution of the linear system. [1]_

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020
    """

    def __init__(self):
        super().__init__(policy=self.__call__, is_deterministic=True)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> Tuple[np.ndarray, Optional["probnum.linalg.solvers.LinearSolverState"]]:
        # Compute residual if necessary
        residual, solver_state = _get_residual(
            problem=problem, belief=belief, solver_state=solver_state
        )

        # A-conjugate search direction / action (assuming exact arithmetic)
        action = -belief.Ainv.mean @ residual

        # Update solver state
        if solver_state is not None:
            solver_state.actions.append(action)
        return action, solver_state
