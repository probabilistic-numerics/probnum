"""Belief update in a matrix-based inference view where the information is given by
matrix-vector multiplication."""
import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import linops, randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief

from ._linear_solver_belief_update import LinearSolverBeliefUpdate


class MatrixBasedLinearBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Gaussian belief update in a matrix-based inference framework for linear information.

    Updates the belief over the quantities of interest of a linear system for a matrix-variate Gaussian belief with Kronecker covariance structure and linear observations.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        # Inference for A
        print(solver_state.belief.A)
        A = self._matrix_based_update(
            matrix=solver_state.belief.A,
            action=solver_state.action,
            observ=solver_state.observation,
        )

        # Inference for Ainv (interpret action and observation as swapped)
        Ainv = self._matrix_based_update(
            matrix=solver_state.belief.Ainv,
            action=solver_state.observation,
            observ=solver_state.action,
        )
        return LinearSystemBelief(A=A, Ainv=Ainv, b=solver_state.belief.b)

    def _matrix_based_update(
        self, matrix: randvars.Normal, action: np.ndarray, observ: np.ndarray
    ) -> randvars.Normal:
        """Matrix-based inference update for linear information."""
        if not isinstance(matrix.cov, linops.Kronecker):
            raise ValueError(
                f"Covariance must have Kronecker structure, but is '{type(matrix.cov).__name__}'."
            )

        pred = matrix.mean @ action
        resid = observ - pred
        covfactor_Ms = matrix.cov.B @ action
        gram = action.T @ covfactor_Ms
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0
        gain = covfactor_Ms * gram_pinv
        covfactor_update = linops.aslinop(gain[:, None]) @ linops.aslinop(
            covfactor_Ms[None, :]
        )
        resid_gain = linops.aslinop(resid[:, None]) @ linops.aslinop(
            gain[None, :]
        )  # residual and gain are flipped due to matrix vectorization

        return randvars.Normal(
            mean=matrix.mean + resid_gain,
            cov=linops.Kronecker(A=matrix.cov.A, B=matrix.cov.B - covfactor_update),
        )
