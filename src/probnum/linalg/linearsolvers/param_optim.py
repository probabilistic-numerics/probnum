"""Hyperparameter optimization routines for probabilistic linear solvers."""

from typing import Callable, Optional, Tuple

import numpy as np

import probnum  # pylint: disable="unused-import
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["HyperparameterOptimization", "UncertaintyCalibration", "OptimalNoiseScale"]


class HyperparameterOptimization:
    """Optimization of hyperparameters of probabilistic linear solvers."""

    def __init__(
        self,
        hyperparam_optim: Callable[
            [
                LinearSystem,
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
            Tuple[
                Tuple[np.ndarray, ...],
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
        ],
    ):
        self._hyperparam_optim = hyperparam_optim

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """Return an action based on the given problem and model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief
            Belief over the solution :math:`x`, the system matrix :math:`A`, its
            inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        optimal_hyperparams
            Optimized hyperparameters.
        belief
            Updated belief over the solution :math:`x`, the system matrix :math:`A`, its
            inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Updated solver state.
        """
        return self._hyperparam_optim(problem, belief, solver_state)


class UncertaintyCalibration(HyperparameterOptimization):
    """Calibrate the uncertainty of the covariance class."""

    def __init__(self):
        super().__init__(hyperparam_optim=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """"""
        raise NotImplementedError


class OptimalNoiseScale(HyperparameterOptimization):
    """Estimate the noise level of a noisy linear system."""

    def __init__(self):
        super().__init__(hyperparam_optim=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """"""
        raise NotImplementedError
