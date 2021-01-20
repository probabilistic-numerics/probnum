"""Abstract base class for belief updates for probabilistic linear solvers."""
import abc
from typing import Optional, Tuple

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum  # pylint: disable="unused-import"
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSolverBeliefUpdate"]

# pylint: disable="invalid-name,too-many-arguments"


class LinearSolverBeliefUpdate(abc.ABC):
    r"""Belief update of a probabilistic linear solver.

    Computes the updated beliefs over quantities of interest of a linear system after
    making observations about the system given a prior belief.

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given a symmetric
        matrix-variate normal belief and linear observations.
    """

    def update(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional["probnum.PNMethodHyperparams"] = None,
        solver_state: Optional["probnum.PNMethodState"] = None,
    ) -> Tuple[
        LinearSystemBelief, Optional["probnum.linalg.solvers.LinearSolverState"]
    ]:
        """Update the belief given observations.

        Parameters
        ----------
        problem :
            Linear system to solve.
        action :
            Action to probe the linear system with.
        observation :
            Observation of the linear system for the given action.
        hyperparams :
            Hyperparameters of the belief.
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
