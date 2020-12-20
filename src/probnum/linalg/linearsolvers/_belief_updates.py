"""Belief updates for probabilistic linear solvers."""

import numpy as np

import probnum  # pylint: disable="unused-import


class BeliefUpdate:
    """Belief update of a probabilistic linear solver.

    Parameters
    ----------
    """

    def __init__(self, belief_update):
        self._belief_update = belief_update

    def __call__(
        self,
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
        action: np.ndarray,
        observation: np.ndarray,
    ):
        """Update belief over quantities of interest of the linear system.

        Parameters
        ----------
        solver_state
        action
        observation
        """
        return self._belief_update(solver_state, action, observation)


class LinearGaussianBeliefUpdate(BeliefUpdate):
    """"""
