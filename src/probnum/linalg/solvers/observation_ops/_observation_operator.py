"""Base class for observation operators of probabilistic linear solvers."""

from typing import Callable, Optional, Tuple

import numpy as np

import probnum  # pylint: disable="unused-import
from probnum.linalg.solvers.data import LinearSolverObservation
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ObservationOperator"]


class ObservationOperator:
    """Observation operator of probabilistic linear solvers.

    Defines the way observations about the linear system are collected for a given
    action.

    Parameters
    ----------
    observation_op
        Callable defining the observation process.

    Examples
    --------

    See Also
    --------
    MatrixMultObservation : Matrix-vector product observations.
    """

    def __init__(
        self,
        observation_op: Callable[
            [
                LinearSystem,
                np.ndarray,
                Optional["probnum.linalg.solvers.LinearSolverState"],
            ],
            LinearSolverObservation,
        ],
    ):
        self._observation_op = observation_op

    def __call__(
        self,
        problem: LinearSystem,
        action: np.ndarray,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> LinearSolverObservation:
        """Collect an observation about the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.
        action :
            Action of the solver to probe the linear system with.
        solver_state :
            Current state of the linear solver.
        """
        return self._observation_op(problem, action, solver_state)
