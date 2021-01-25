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
        data: "probnum.linalg.solvers.data.LinearSolverData",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[Union[np.ndarray, float], ...],
        Optional["probnum.linalg.solvers.LinearSolverState"],
    ]:
        # TODO initialize solver state if None
        raise NotImplementedError

    @staticmethod
    def _optimal_noise_scale_iterative(
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> float:
        n = solver_state.problem.A.shape[0]
        k = solver_state.info.iteration
        sum_delta_invgram_delta = solver_state.cache.A.sum_delta_invgram_delta
        return np.maximum(0.0, sum_delta_invgram_delta / (n * k) - 1)

    @staticmethod
    def _optimal_noise_scale_batch(
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> float:
        # Compute intermediate quantities
        Delta0 = (
            solver_state.data.observations_arr
            - solver_state.prior.A.mean @ solver_state.data.actions_arr
        )
        SW0S = solver_state.data.actions_arr.T @ (
            solver_state.prior.A.cov.A @ solver_state.data.actions_arr
        )
        try:
            SW0SinvSDelta0 = scipy.linalg.solve(
                SW0S, solver_state.data.actions_arr.T @ Delta0, assume_a="pos"
            )  # solves k x k system k times: O(k^3)
            linop_rhs = Delta0.T @ (
                2 * solver_state.prior.A.cov.A.inv() @ Delta0
                - solver_state.data.actions_arr @ SW0SinvSDelta0
            )
            linop_tracearg = scipy.linalg.solve(
                SW0S, linop_rhs, assume_a="pos"
            )  # solves k x k system k times: O(k^3)

            # Optimal noise scale with respect to the evidence
            noise_scale_estimate = (
                linop_tracearg.trace()
                / (
                    solver_state.problem.A.shape[0]
                    * solver_state.data.actions_arr.shape[1]
                )
                - 1
            )
        except scipy.linalg.LinAlgError as err:
            raise scipy.linalg.LinAlgError(
                "Matrix S'W_0S not invertible. Noise scale estimate may be inaccurate."
            ) from err

        return noise_scale_estimate if noise_scale_estimate > 0.0 else 0.0
