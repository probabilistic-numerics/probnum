"""Linear system belief updates.

Class defining how a belief over quantities of interest of a linear system is updated
given information about the problem.
"""
import probnum  # pylint: disable="unused-import"
from probnum import randvars


class LinearSystemBeliefUpdate:
    r"""Belief update for the quantities of interest of a linear system.

    Given a solver state containing information about the linear system collected in the current step, update the belief about the quantities of interest.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> "probnum.linalg.solvers.beliefs.LinearSystemBelief":
        r"""Update the belief about the quantities of interest of a linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        pass

    def _update_x(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> randvars.RandomVariable:
        """Update the belief about the solution."""
        raise NotImplementedError

    def _update_A(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> randvars.RandomVariable:
        """Update the belief about the matrix."""
        raise NotImplementedError

    def _update_Ainv(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> randvars.RandomVariable:
        """Update the belief about the (pseudo-)inverse of the system matrix."""
        raise NotImplementedError

    def _update_b(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> randvars.RandomVariable:
        """Update the belief about the right hand side."""
        raise NotImplementedError
