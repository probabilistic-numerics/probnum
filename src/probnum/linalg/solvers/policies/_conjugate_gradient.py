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
        Reorthogonalization function, which takes a vector, an orthogonal basis and optionally an inner product and returns an orthogonal vector.
    reorthogonalization_target
        Vector to reorthogonalize. Either the current `action` or `residual`.
    """

    def __init__(
        self,
        reorthogonalization_fn: Optional[
            Callable[
                [np.ndarray, Iterable[np.ndarray], linops.LinearOperator], np.ndarray
            ]
        ] = None,
        reorthogonalization_target: str = "residual",
    ) -> None:
        self._reorthogonalization_fn = reorthogonalization_fn
        self._reorthogonalization_target = reorthogonalization_target

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> np.ndarray:

        action = -solver_state.residual.copy()

        if solver_state.step > 0:
            # Reorthogonalization of the residual
            if (
                self._reorthogonalization_target == "residual"
                and self._reorthogonalization_fn is not None
            ):
                residual = None
                prev_residual = None
            else:
                residual = solver_state.residual
                prev_residual = solver_state.residuals[solver_state.step - 1]

            # A-conjugacy correction (in exact arithmetic)
            beta = (np.linalg.norm(residual) / np.linalg.norm(prev_residual)) ** 2
            action += beta * solver_state.actions[solver_state.step - 1]

            # Reorthogonalization of the resulting action
            if (
                self._reorthogonalization_target == "action"
                and self._reorthogonalization_fn is not None
            ):
                return self._reorthogonalized_action(
                    action=action, solver_state=solver_state
                )

        return action

    def _reorthogonalized_action(
        self,
        action: np.ndarray,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> np.ndarray:
        if isinstance(solver_state.prior.x, randvars.Normal):
            inprod_matrix = (
                solver_state.problem.A
                @ solver_state.prior.x.cov
                @ solver_state.problem.A.T
            )
        elif isinstance(solver_state.prior.x, randvars.Constant):
            inprod_matrix = solver_state.problem.A

        orthogonal_basis = np.asarray(solver_state.actions[0 : solver_state.step])

        return self._reorthogonalization_fn(
            action,
            orthogonal_basis,
            inprod_matrix,
        )
