"""Abstract base class for yperparameter optimization of probabilistic linear
solvers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

import probnum  # pylint: disable="unused-import
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["HyperparameterOptimization"]

# pylint: disable="invalid-name"


class HyperparameterOptimization(ABC):
    """Optimization of hyperparameters of probabilistic linear solvers."""

    @abstractmethod
    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[Union[np.ndarray, float], ...],
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """Optimized hyperparameters of the linear system model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        actions :
            Actions of the solver to probe the linear system with.
        observations :
            Observations of the linear system for the given actions.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        optimal_hyperparams
            Optimized hyperparameters.
        belief
            Updated belief about the solution :math:`x`, the system matrix :math:`A`, its
            inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Updated solver state.
        """
        raise NotImplementedError
