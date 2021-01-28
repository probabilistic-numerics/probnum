import math
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy

import probnum
import probnum.linops as linops
from probnum.linalg.solvers._state import LinearSolverState
from probnum.linalg.solvers.belief_updates._symmetric_normal_linear_obs import (
    _SymmetricNormalLinearObsCache,
)
from probnum.linalg.solvers.hyperparam_optim._hyperparameter_optimization import (
    HyperparameterOptimization,
)
from probnum.linalg.solvers.hyperparams import LinearSystemNoise
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["OptimalNoiseScale"]


class OptimalNoiseScale(HyperparameterOptimization):
    r"""Estimate the noise level of a noisy linear system.

    Computes the optimal noise scale maximizing the log-marginal likelihood.

    Parameters
    ----------
    iterative :
        Compute the optimal noise scale in an iterative fashion or recompute it from
        the entire batch of actions and observations.
    """

    def __init__(self, iterative: bool = True):
        self.iterative = iterative
        super().__init__()

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        data: "probnum.linalg.solvers.data.LinearSolverData",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> LinearSystemNoise:
        if solver_state is None:
            solver_state = LinearSolverState(
                problem=problem,
                belief=belief,
                data=data,
                cache=_SymmetricNormalLinearObsCache(
                    problem=problem, belief=belief, data=data
                ),
            )

        if self.iterative:
            eps_sq = self._compute_iterative(solver_state=solver_state)
        else:
            eps_sq = self._compute_batch(solver_state=solver_state)

        return LinearSystemNoise(
            epsA_cov=linops.SymmetricKronecker(
                A=math.sqrt(eps_sq) * solver_state.prior.A.cov.A
            )
        )

    @staticmethod
    def _compute_iterative(
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> float:
        n = solver_state.problem.A.shape[0]
        k = solver_state.info.iteration
        sum_delta_invgram_delta = solver_state.cache.sum_deltaA_invgramA_deltaA
        return np.maximum(0.0, sum_delta_invgram_delta / (n * k) - 1)

    @staticmethod
    def _compute_batch(
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> float:
        # Compute intermediate quantities
        Delta0 = (
            solver_state.data.observations_arr.obsA
            - solver_state.prior.A.mean @ solver_state.data.actions_arr.actA
        )
        SW0S = solver_state.data.actions_arr.actA.T @ (
            solver_state.prior.A.cov.A @ solver_state.data.actions_arr.actA
        )
        try:
            SW0SinvSDelta0 = scipy.linalg.solve(
                SW0S, solver_state.data.actions_arr.actA.T @ Delta0, assume_a="pos"
            )
            linop_rhs = Delta0.T @ (
                2 * solver_state.prior.A.cov.A.inv() @ Delta0
                - solver_state.data.actions_arr.actA @ SW0SinvSDelta0
            )
            linop_tracearg = scipy.linalg.solve(SW0S, linop_rhs, assume_a="pos")
        except scipy.linalg.LinAlgError as err:
            raise scipy.linalg.LinAlgError(
                "Matrix S'W_0S not invertible. Noise scale estimate may be inaccurate."
            ) from err

        # Optimal noise scale with respect to the evidence
        noise_scale_estimate = (
            linop_tracearg.trace()
            / (
                solver_state.problem.A.shape[0]
                * solver_state.data.actions_arr.actA.shape[1]
            )
            - 1
        )

        return np.maximum(0.0, noise_scale_estimate)
