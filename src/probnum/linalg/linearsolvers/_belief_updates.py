"""Belief updates for probabilistic linear solvers."""

import numpy as np

import probnum  # pylint: disable="unused-import


class BeliefUpdate:
    """Belief update of a probabilistic linear solver.

    Computes a new belief over the quantities of interest of the linear system based
    on the current state of the linear solver.

    Parameters
    ----------
    belief_update
        Callable defining how to update the belief.

    Examples
    --------

    See Also
    --------
    LinearGaussianBeliefUpdate: Belief update given linear observations.
    """

    def __init__(self, belief_update):
        self._belief_update = belief_update

    def __call__(self, solver_state: "probnum.linalg.linearsolvers.LinearSolverState"):
        """Update belief over quantities of interest of the linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        return self._belief_update(solver_state)


class LinearGaussianBeliefUpdate(BeliefUpdate):
    """"""

    def __init__(self):
        super().__init__(belief_update=self.__call__)

    def __call__(self, solver_state: "probnum.linalg.linearsolvers.LinearSolverState"):
        """Belief update assuming Gaussianity and linear observations.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
