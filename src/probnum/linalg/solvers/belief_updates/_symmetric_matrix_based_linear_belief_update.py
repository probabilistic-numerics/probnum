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
        print(solver_state.belief.A)
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
        return LinearSystemBelief(A=A, Ainv=Ainv, b=solver_state.belief.b)

    def _symmetric_matrix_based_update(
        self, matrix: randvars.Normal, action: np.ndarray, observ: np.ndarray
    ):
        """Symmetric matrix-based inference update for linear information."""
        if not isinstance(matrix.cov, linops.SymmetricKronecker):
            raise ValueError(
                f"Covariance must have symmetric Kronecker structure, but is '{type(matrix.cov).__name__}'."
            )
