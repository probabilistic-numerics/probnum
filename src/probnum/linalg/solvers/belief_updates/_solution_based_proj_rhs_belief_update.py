"""Belief update in a solution-based inference view where the information is given by
projecting the current residual to a subspace."""
import numpy as np

import probnum  # pylint: disable="unused-import"
import probnum.randvars as randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.typing import FloatArgType

from ._linear_solver_belief_update import LinearSolverBeliefUpdate


class SolutionBasedProjectedRHSBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Gaussian belief update in a solution-based framework for projected right-hand side information.

    Updates the belief over the quantities of interest of a linear system for a Gaussian
    belief over the solution :math:`x` of the linear system :math:`Ax=b` given information of the form :math:`y = b^\top s=(Ax)^\top s`.

    Parameters
    ----------
    noise_var :
        Scalar observation noise.
    """

    def __init__(self, noise_var: FloatArgType = 0.0) -> None:
        self._noise_var = noise_var

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        A_action = solver_state.problem.A @ solver_state.action
        pred = A_action.T @ solver_state.belief.x.mean
        resid = solver_state.observation - pred
        cov_xy = solver_state.belief.x.cov @ A_action
        gram = A_action.T @ cov_xy + self._noise_var
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0
        gain = cov_xy * gram_pinv
        cov_update = gain @ cov_xy.T

        x = randvars.Normal(
            mean=solver_state.belief.x.mean + gain * resid,
            cov=solver_state.belief.x.cov - cov_update,
        )
        Ainv = solver_state.belief.Ainv + cov_update

        return LinearSystemBelief(
            x=x, A=solver_state.belief.A, Ainv=Ainv, b=solver_state.belief.b
        )