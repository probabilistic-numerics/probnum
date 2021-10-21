"""Linear system belief updates.

Class defining how a belief over quantities of interest of a linear system is updated
given information about the problem.
"""
import abc

import probnum  # pylint: disable="unused-import"
from probnum.linalg.solvers.beliefs import LinearSystemBelief


class LinearSolverBeliefUpdate(abc.ABC):
    r"""Belief update for the quantities of interest of a linear system.

    Given a solver state containing information about the linear system collected in the current step, update the belief about the quantities of interest.
    """

    @abc.abstractmethod
    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:
        r"""Update the belief about the quantities of interest of a linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
