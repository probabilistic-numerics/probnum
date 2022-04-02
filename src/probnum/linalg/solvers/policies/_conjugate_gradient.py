"""Policy returning :math:`A`-conjugate actions."""

from typing import Callable, Iterable, Optional, Tuple

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import linops, randvars

from . import _linear_solver_policy


class ConjugateGradientPolicy(_linear_solver_policy.LinearSolverPolicy):
    r"""Policy returning :math:`A`-conjugate actions.

    Selects the negative gradient / residual as an initial action
    :math:`s_0 = b - A x_0` and then successively generates :math:`A`-conjugate
    actions, i.e. the actions satisfy :math:`s_i^\top A s_j = 0` iff
    :math:`i \neq j`.

    Parameters
    ----------
    reorthogonalization_fn_residual
        Reorthogonalization function, which takes a vector, an orthogonal basis
        and optionally an inner product and returns a reorthogonalized vector. If
        not `None` the residuals are reorthogonalized before the action is computed.
    reorthogonalization_fn_action
        Reorthogonalization function, which takes a vector, an orthogonal basis
        and optionally an inner product and returns a reorthogonalized vector.
        If not `None` the computed action is reorthogonalized.
    """

    def __init__(
        self,
        reorthogonalization_fn_residual: Optional[
            Callable[
                [np.ndarray, Iterable[np.ndarray], linops.LinearOperator], np.ndarray
            ]
        ] = None,
        reorthogonalization_fn_action: Optional[
            Callable[
                [np.ndarray, Iterable[np.ndarray], linops.LinearOperator], np.ndarray
            ]
        ] = None,
    ) -> None:
        self._reorthogonalization_fn_residual = reorthogonalization_fn_residual
        self._reorthogonalization_fn_action = reorthogonalization_fn_action

    def __call__(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:

        residual = solver_state.residual

        if solver_state.step == 0:
            if self._reorthogonalization_fn_residual is not None:
                solver_state.cache["reorthogonalized_residuals"].append(
                    solver_state.residual
                )

            return residual
        else:
            # Reorthogonalization of the residual
            if self._reorthogonalization_fn_residual is not None:
                residual, prev_residual = self._reorthogonalized_residual(
                    solver_state=solver_state
                )
            else:
                prev_residual = solver_state.residuals[solver_state.step - 1]

            # A-conjugacy correction (in exact arithmetic)
            beta = (np.linalg.norm(residual) / np.linalg.norm(prev_residual)) ** 2
            action = residual + beta * solver_state.actions[solver_state.step - 1]

            # Reorthogonalization of the resulting action
            if self._reorthogonalization_fn_action is not None:
                action = self._reorthogonalized_action(
                    action=action, solver_state=solver_state
                )

            return action

    def _reorthogonalized_residual(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the reorthogonalized residual and its predecessor."""
        residual = self._reorthogonalization_fn_residual(
            v=solver_state.residual,
            orthogonal_basis=np.asarray(
                solver_state.cache["reorthogonalized_residuals"]
            ),
            inner_product=None,
        )
        solver_state.cache["reorthogonalized_residuals"].append(residual)
        prev_residual = solver_state.cache["reorthogonalized_residuals"][
            solver_state.step - 1
        ]
        return residual, prev_residual

    def _reorthogonalized_action(
        self,
        action: np.ndarray,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> np.ndarray:
        """Reorthogonalize the computed action."""
        if isinstance(solver_state.prior.x, randvars.Normal):
            inprod_matrix = (
                solver_state.problem.A
                @ solver_state.prior.x.cov
                @ solver_state.problem.A.T
            )
        elif isinstance(solver_state.prior.x, randvars.Constant):
            inprod_matrix = solver_state.problem.A

        orthogonal_basis = np.asarray(solver_state.actions[0 : solver_state.step])

        return self._reorthogonalization_fn_action(
            v=action,
            orthogonal_basis=orthogonal_basis,
            inner_product=inprod_matrix,
        )
