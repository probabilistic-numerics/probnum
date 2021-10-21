"""Belief update in a matrix-based inference view assuming symmetry where the
information is given by matrix-vector multiplication."""
import probnum  # pylint: disable="unused-import"
import probnum.randvars as randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief

from ._linear_solver_belief_update import LinearSolverBeliefUpdate


class SymmetricMatrixBasedLinearBeliefUpdate(LinearSolverBeliefUpdate):
    r""""""

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        A = self._update_A(solver_state=solver_state)

        Ainv = self._update_Ainv(solver_state=solver_state)
        b = solver_state.belief.b
        return LinearSystemBelief(A=A, Ainv=Ainv, b=b)

    def _update_A(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> randvars.RandomVariable:
        raise NotImplementedError

    def _update_Ainv(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> randvars.RandomVariable:
        raise NotImplementedError
