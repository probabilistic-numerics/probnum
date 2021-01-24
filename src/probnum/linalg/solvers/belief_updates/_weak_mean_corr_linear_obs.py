"""Belief update for the weak mean correspondence belief given linear observations."""

from functools import cached_property
from typing import Optional, Tuple

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers.belief_updates._symmetric_normal_linear_obs import (
    SymmetricNormalLinearObsBeliefUpdate,
    _InverseMatrixSymmetricNormalLinearObsBeliefUpdate,
    _SystemMatrixSymmetricNormalLinearObsBeliefUpdate,
)
from probnum.linalg.solvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.linalg.solvers.hyperparams import (
    LinearSolverHyperparams,
    UncertaintyUnexploredSpace,
)
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["WeakMeanCorrLinearObsBeliefUpdate"]


class _SystemMatrixWeakMeanCorrLinearObsBeliefUpdateState(
    _SystemMatrixSymmetricNormalLinearObsBeliefUpdate
):
    """Weak mean correspondence belief update for the system matrix."""

    # TODO use assumptions WS = Y and WY=H_0Y (Theorem 3, eqn. 1+2, Wenger2020)
    # @cached_property
    # def covfactor_action(self) -> np.ndarray:
    #     return self.qoi_belief.cov.A @ self.action
    #
    # @cached_property
    # def action_covfactor_action(self) -> float:
    #     return self.action.T @ self.covfactor_action

    def __call__(
        self,
        belief: WeakMeanCorrespondenceBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[UncertaintyUnexploredSpace] = None,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> rvs.Normal:
        """Updated belief for the system matrix."""

        # Action observation inner products
        # TODO group all precomputed quantities in LinearSolverMiscQuantities
        #   and move all belief update related functions back into belief update.
        if self.solver_state is not None:
            try:
                action_obs_innerprod = self.solver_state.action_obs_innerprods[
                    self.solver_state.iteration
                ]
            except IndexError:
                action_obs_innerprod = self.action.T @ self.observation
        else:
            action_obs_innerprod = self.action.T @ self.observation
        # TODO
        covfactor_op = (
            self.belief._cov_factor_matrix(action_obs_innerprods=action_obs_innerprod)
            - self.covfactor_updates_batch[0]
        )

        mean = linops.aslinop(self.prior.A.mean) + self.mean_update_batch
        cov = linops.SymmetricKronecker(covfactor_op)
        return rvs.Normal(mean=mean, cov=cov)


class _InverseMatrixWeakMeanCorrLinearObsBeliefUpdateState(
    _InverseMatrixSymmetricNormalLinearObsBeliefUpdate
):
    """Weak mean correspondence belief update for the inverse."""

    def __call__(
        self,
        belief: WeakMeanCorrespondenceBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[UncertaintyUnexploredSpace] = None,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> rvs.Normal:
        """Updated belief for the inverse."""
        # Empirical prior with scaled uncertainty in null space of observations

        # TODO
        covfactor_op = (
            self.belief._cov_factor_inverse(hyperparams=hyperparams)
            - self.covfactor_updates_batch[0]
        )

        # Recursive trace update (See Section S4.3 of Wenger and Hennig, 2020)
        covfactor_op.trace = (
            self.belief.Ainv.cov.A.trace()
            - self.sqnorm_covfactor_action / self.action_covfactor_action
        )

        mean = linops.aslinop(self.prior.Ainv.mean) + self.mean_update_batch
        cov = linops.SymmetricKronecker(covfactor_op)
        return rvs.Normal(mean=mean, cov=cov)


class WeakMeanCorrLinearObsBeliefUpdate(SymmetricNormalLinearObsBeliefUpdate):
    r"""Weak mean correspondence belief update assuming linear observations."""

    def __init__(self, problem: LinearSystem, prior: LinearSystemBelief):
        super().__init__(
            problem=problem,
            prior=prior,
            A_belief_update_type=_SystemMatrixWeakMeanCorrLinearObsBeliefUpdateState,
            Ainv_belief_update_type=_InverseMatrixWeakMeanCorrLinearObsBeliefUpdateState,
        )

    def __init__(
        self,
        belief: LinearSystemBelief,
        action: "probnum.linalg.solvers.data.LinearSolverAction",
        observation: "probnum.linalg.solvers.data.LinearSolverObservation",
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams"
        ] = None,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> Tuple[
        LinearSystemBelief, Optional["probnum.linalg.solvers.LinearSolverState"]
    ]:
        pass
        # TODO modify empirical prior with actions and observations here in solver
        #  state??
