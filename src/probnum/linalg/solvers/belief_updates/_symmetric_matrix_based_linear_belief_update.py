"""Belief update in a matrix-based inference view assuming symmetry where the
information is given by matrix-vector multiplication."""
import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import linops, randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief

from ._linear_solver_belief_update import LinearSolverBeliefUpdate


class SymmetricMatrixBasedLinearBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Symmetric Gaussian belief update in a matrix-based inference framework for linear information.

    Updates the belief over the quantities of interest of a linear system for a symmetric matrix-variate Gaussian belief and linear observations.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        # Inference for A
        A = self._symmetric_matrix_based_update(
            matrix=solver_state.belief.A,
            action=solver_state.action,
            observ=solver_state.observation,
        )

        # Inference for Ainv (interpret action and observation as swapped)
        Ainv = self._symmetric_matrix_based_update(
            matrix=solver_state.belief.Ainv,
            action=solver_state.observation,
            observ=solver_state.action,
        )
        return LinearSystemBelief(A=A, Ainv=Ainv, x=None, b=solver_state.belief.b)

    def _symmetric_matrix_based_update(
        self, matrix: randvars.Normal, action: np.ndarray, observ: np.ndarray
    ) -> randvars.Normal:
        """Symmetric matrix-based inference update for linear information."""
        if not isinstance(matrix.cov, linops.SymmetricKronecker):
            raise ValueError(
                f"Covariance must have symmetric Kronecker structure, but is '{type(matrix.cov).__name__}'."
            )

        pred = matrix.mean @ action
        resid = observ - pred
        covfactor_Ms = matrix.cov.A @ action
        gram = action.T @ covfactor_Ms
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0
        gain = covfactor_Ms * gram_pinv
        covfactor_update = gain @ covfactor_Ms.T
        resid_gain = linops.aslinop(resid[:, None]) @ linops.aslinop(gain[None, :])

        return randvars.Normal(
            mean=matrix.mean
            + resid_gain
            + resid_gain.T
            - linops.aslinop(gain[:, None])
            @ linops.aslinop((action.T @ resid_gain)[None, :]),
            cov=linops.SymmetricKronecker(A=matrix.cov.A - covfactor_update),
        )
