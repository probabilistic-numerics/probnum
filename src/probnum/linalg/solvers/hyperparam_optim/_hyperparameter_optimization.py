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
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        data: "probnum.linalg.solvers.data.LinearSolverData",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams":
        """Optimized hyperparameters of the linear system model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        data :
            Actions and corresponding observations of the probabilistic linear solver.
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
