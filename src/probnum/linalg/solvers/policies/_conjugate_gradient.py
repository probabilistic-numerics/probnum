"""Policy returning :math:`A`-conjugate actions."""

import numpy as np

import probnum  # pylint: disable="unused-import"

from . import _linear_solver_policy


class ConjugateGradientPolicy(_linear_solver_policy.LinearSolverPolicy):
    r"""Policy returning :math:`A`-conjugate actions.

    Selects the negative gradient / residual as an initial action :math:`s_0 = b - A x_0` and then successively generates :math:`A`-conjugate actions, i.e. the actions satisfy :math:`s_i^\top A s_j = 0` iff :math:`i \neq j`.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> np.ndarray:

        action = -solver_state.residual.copy()

        if solver_state.step > 0:
            # A-conjugacy correction (in exact arithmetic)
            beta = (
                np.linalg.norm(solver_state.residual)
                / np.linalg.norm(solver_state.residuals[solver_state.step - 1])
            ) ** 2

            action += beta * solver_state.actions[solver_state.step - 1]

        return action
