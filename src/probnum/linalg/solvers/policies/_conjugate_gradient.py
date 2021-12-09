"""Policy returning :math:`A`-conjugate actions."""

from typing import Callable, Iterable, Optional

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import linops, randvars

from . import _linear_solver_policy


class ConjugateGradientPolicy(_linear_solver_policy.LinearSolverPolicy):
    r"""Policy returning :math:`A`-conjugate actions.

    Selects the negative gradient / residual as an initial action :math:`s_0 = b - A x_0` and then successively generates :math:`A`-conjugate actions, i.e. the actions satisfy :math:`s_i^\top A s_j = 0` iff :math:`i \neq j`.

    Parameters
    ----------
    reorthogonalization_fn
        Reorthogonalization function, which takes a vector an orthogonal basis and optionally an inner product and returns a reorthogonalized vector.
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

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> np.ndarray:

        action = -solver_state.residual.copy()

        if solver_state.step > 0:
            # A-conjugacy correction (in exact arithmetic)
            beta = (
                np.linalg.norm(solver_state.residual)
                / np.linalg.norm(solver_state.residuals[solver_state.step - 1])
            ) ** 2

            action += beta * solver_state.actions[solver_state.step - 1]

            # (Optional) Reorthogonalization
            if self._reorthogonalization_fn is not None:
                if isinstance(solver_state.prior.x, randvars.Normal):
                    inprod_matrix = (
                        solver_state.problem.A
                        @ solver_state.prior.x.cov
                        @ solver_state.problem.A.T
                    )
                elif isinstance(solver_state.prior.x, randvars.Constant):
                    inprod_matrix = solver_state.problem.A

                action = self._reorthogonalization_fn(
                    action,
                    solver_state.prev_actions,
                    inprod_matrix,
                )

        return action
