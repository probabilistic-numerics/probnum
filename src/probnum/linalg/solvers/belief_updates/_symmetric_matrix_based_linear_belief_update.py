"""Belief update in a matrix-based inference view assuming symmetry where the
information is given by matrix-vector multiplication."""

import probnum  # pylint: disable="unused-import"
from probnum.linalg.solvers.beliefs import LinearSystemBelief

from ._linear_solver_belief_update import LinearSolverBeliefUpdate


class SymmetricMatrixBasedLinearBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Symmetric Gaussian belief update in a matrix-based inference framework for linear information.

    Updates the belief over the quantities of interest of a linear system for a symmetric matrix-variate Gaussian belief and linear observations.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        A = None
        Ainv = None

        return LinearSystemBelief(A=A, Ainv=Ainv, b=solver_state.belief.b)
