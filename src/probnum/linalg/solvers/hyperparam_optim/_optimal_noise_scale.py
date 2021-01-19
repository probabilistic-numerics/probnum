from typing import List, Optional, Tuple, Union

import numpy as np
import scipy

import probnum
from probnum.linalg.solvers.hyperparam_optim._hyperparameter_optimization import (
    HyperparameterOptimization,
)
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["OptimalNoiseScale"]


class OptimalNoiseScale(HyperparameterOptimization):
    r"""Estimate the noise level of a noisy linear system.

    Computes the optimal noise scale maximizing the log-marginal likelihood.
    """

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[Union[np.ndarray, float], ...],
        Optional["probnum.linalg.solvers.LinearSolverState"],
    ]:

        raise NotImplementedError

    @staticmethod
    def _optimal_noise_scale_iterative(
        previous_optimal_noise_scale: float,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def _optimal_noise_scale_batch(
        problem: LinearSystem,
        prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        actions: np.ndarray,
        observations: np.ndarray,
    ) -> float:
        # Compute intermediate quantities
        Delta0 = observations - prior.A.mean @ actions
        SW0S = actions.T @ (prior.A.cov.A @ actions)
        try:
            SW0SinvSDelta0 = scipy.linalg.solve(
                SW0S, actions.T @ Delta0, assume_a="pos"
            )  # solves k x k system k times: O(k^3)
            linop_rhs = Delta0.T @ (
                2 * prior.A.cov.A.inv() @ Delta0 - actions @ SW0SinvSDelta0
            )
            linop_tracearg = scipy.linalg.solve(
                SW0S, linop_rhs, assume_a="pos"
            )  # solves k x k system k times: O(k^3)

            # Optimal noise scale with respect to the evidence
            noise_scale_estimate = (
                linop_tracearg.trace() / (problem.A.shape[0] * actions.shape[1]) - 1
            )
        except scipy.linalg.LinAlgError as err:
            raise scipy.linalg.LinAlgError(
                "Matrix S'W_0S not invertible. Noise scale estimate may be inaccurate."
            ) from err

        return noise_scale_estimate if noise_scale_estimate > 0.0 else 0.0
