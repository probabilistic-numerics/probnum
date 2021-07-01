"""Policy returning :math:`A`-conjugate actions."""
from typing import Callable, Iterable, Optional

import numpy as np

import probnum as pn
from probnum import linops

from . import _policy


class ConjugateGradient(_policy.Policy):
    r"""Policy returning :math:`A`-conjugate actions.

    Selects the negative gradient / residual as an initial action :math:`s_0 = A x_0 - b` and then successively generates :math:`A`-conjugate actions, i.e. the actions satisfy :math:`s_i^\top A s_j = 0` iff :math:`i \neq j`.

    Parameters
    ----------
    reorthogonalization_fn :
        Callable which reorthogonalizes a vector with respect to a set of vectors with respect to an inner product defined by a linear operator.
    """

    def __init__(
        self,
        reorthogonalization_fn: Optional[
            Callable[
                [np.ndarray, Iterable[np.ndarray], linops.LinearOperator], np.ndarray
            ]
        ] = None,
    ) -> None:
        self._reorthogonalization_fn = reorthogonalization_fn

    def __call__(self, solver_state: "pn.linalg.solvers.State") -> np.ndarray:

        action = solver_state.residual.copy()

        if solver_state.iteration > 0:
            # A-conjugate action (in exact arithmetic)
            beta = (
                solver_state.residual_norm_squared
                / solver_state.prev_residual_norm_squared
            )

            action += beta * solver_state.actions[-1]

            # (Optional) Reorthogonalization
            if self._reorthogonalization_fn is not None:
                action = self._reorthogonalization_fn(
                    action,
                    solver_state.actions,
                    solver_state.problem.A,
                )

        return action
