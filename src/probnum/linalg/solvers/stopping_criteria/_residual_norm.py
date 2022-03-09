"""Stopping criterion based on the norm of the residual."""

import numpy as np

import probnum
from probnum.typing import ScalarLike

from ._linear_solver_stopping_criterion import LinearSolverStoppingCriterion


class ResidualNormStoppingCriterion(LinearSolverStoppingCriterion):
    r"""Residual stopping criterion.

    Terminate when the euclidean norm of the residual :math:`r_{i} = b - A x_{i}` is
    sufficiently small, i.e. if it satisfies :math:`\lVert r_i \rVert_2 \leq \max(
    \text{atol}, \text{rtol} \lVert b \rVert_2)`.

    Parameters
    ----------
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    """

    def __init__(
        self,
        atol: ScalarLike = 10**-5,
        rtol: ScalarLike = 10**-5,
    ):
        self.atol = probnum.utils.as_numpy_scalar(atol)
        self.rtol = probnum.utils.as_numpy_scalar(rtol)

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> bool:
        """Check whether the residual norm is smaller than the specified tolerance.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        residual_norm = np.linalg.norm(solver_state.residual, ord=2)
        b_norm = np.linalg.norm(solver_state.problem.b, ord=2)
        return residual_norm <= self.atol or residual_norm <= self.rtol * b_norm
