"""Belief update given symmetric matrix-variate normal beliefs and linear
observations."""

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import dataclasses
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers._state import LinearSolverCache
from probnum.linalg.solvers.belief_updates import (
    LinearSolverBeliefUpdate,
    LinearSolverQoIBeliefUpdate,
)
from probnum.linalg.solvers.beliefs import (
    LinearSystemBelief,
    SymmetricNormalLinearSystemBelief,
)
from probnum.linalg.solvers.hyperparams import LinearSystemNoise
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"

# Public classes and functions. Order is reflected in documentation.
__all__ = ["SymmetricNormalLinearObsBeliefUpdate"]


@dataclasses.dataclass
class _SymmetricNormalLinearObsCache(LinearSolverCache):
    """Cached quantities assuming symmetric matrix-variate normal priors and linear
    observations."""

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSystemNoise"
        ] = None,
        data: Optional["probnum.linalg.solvers.data.LinearSolverData"] = None,
        prev_cache: Optional["_SymmetricNormalLinearObsCache"] = None,
    ):
        # pylint: disable="too-many-arguments"
        super().__init__(
            problem=problem,
            prior=prior,
            belief=belief,
            hyperparams=hyperparams,
            data=data,
            prev_cache=prev_cache,
        )

    @cached_property
    def action_observation(self) -> float:
        r"""Inner product :math:`s_i^\top y_i` between the current action and
        observation."""
        return (self.action.actA.T @ self.observation.obsA).item()

    @cached_property
    def action_observation_innerprod_list(self) -> List[float]:
        """Inner products :math:`s_i^\top y_i` between action and observation pairs."""
        if self.prev_cache is None:
            if self.data is None:
                return []
            return [self.action_observation]
        else:
            return self.prev_cache.action_observation_innerprod_list + [
                self.action_observation
            ]

    @cached_property
    def log_rayleigh_quotient(self) -> float:
        r"""Log-Rayleigh quotient :math:`\ln R(A, s_i) = \ln(s_i^\top y_i) - \ln (
        s_i^\top s_i)` of the current action and observation."""
        return (
            np.log(self.action_observation)
            - np.log((self.action.actA.T @ self.action.actA)).item()
        )

    @cached_property
    def log_rayleigh_quotient_list(self) -> List[float]:
        r"""Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(
        s_i^\top s_i)`."""
        if self.prev_cache is None:
            if len(self.data) == 1:
                return [self.log_rayleigh_quotient]
            else:
                return list(
                    np.log(
                        np.einsum(
                            "nk,nk->k",
                            self.data.actions_arr.actA,
                            self.data.observations_arr.obsA,
                        )
                    )
                    - np.log(
                        np.einsum(
                            "nk,nk->k",
                            self.data.actions_arr.actA,
                            self.data.actions_arr.actA,
                        )
                    )
                )
        return self.prev_cache.log_rayleigh_quotient_list + [self.log_rayleigh_quotient]

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r = A x_i- b` of the solution estimate
        :math:`x_i=\mathbb{E}[\mathsf{x}]` at iteration :math:`i`."""
        if isinstance(self.hyperparams, LinearSystemNoise):
            if isinstance(self.problem.A, linops.LinearOperator):
                A = self.problem.A
            elif isinstance(self.problem.A, rvs.RandomVariable):
                A = self.problem.A.sample()
            else:
                raise NotImplementedError
            return A @ self.belief.x.mean - rvs.asrandvar(self.problem.b).sample()
        elif self.prev_cache is None:
            return self.problem.A @ self.belief.x.mean - self.problem.b
        else:
            return self.prev_cache.residual + self.step_size * self.observation.obsA

    @cached_property
    def step_size(self) -> np.ndarray:
        r"""Step size :math:`\alpha_i` of the solver viewed as a quadratic optimizer
        taking steps :math:`x_{i+1} = x_i + \alpha_i s_i`."""
        if self.hyperparams is None:
            return (
                -self.action.actA.T @ self.prev_cache.residual / self.action_observation
            ).item()
        elif isinstance(self.hyperparams.A_eps.cov.A, linops.ScalarMult):
            eps_sq = self.hyperparams.A_eps.cov.A.scalar
            return (
                -self.action.actA.T @ self.prev_cache.residual / self.action_observation
            ).item() / (1 + eps_sq)

        else:
            raise NotImplementedError

    @cached_property
    def deltaA(self) -> np.ndarray:
        r"""Residual :math:`\Delta^A_i = y_i - A_{i-1}s_i` between observation and
        prediction."""
        return self.observation.obsA - self.belief.A.mean @ self.action.actA

    @cached_property
    def deltaH(self) -> np.ndarray:
        r"""Residual :math:`\Delta^H_i = s_i - H_{i-1}y_i` between inverse
        observation and prediction."""
        return self.action.actA - self.belief.Ainv.mean @ self.observation.obsA

    @cached_property
    def deltaA_action(self) -> float:
        r"""Inner product :math:`(\Delta^A)^\top s` between matrix residual and
        action."""
        return self.deltaA.T @ self.action.actA

    @cached_property
    def deltaH_observation(self) -> float:
        r"""Inner product :math:`(\Delta^H)^\top y` between inverse residual and
        observation."""
        return self.deltaH.T @ self.observation.obsA

    @cached_property
    def covfactorA_action(self) -> np.ndarray:
        r"""Uncertainty about the matrix along the current action.

        Computes the matrix-vector product :math:`W^A_{i-1}s_i` between the covariance
        factor of the matrix model and the current action.
        """
        return self.belief.A.cov.A @ self.action.actA

    @cached_property
    def covfactorH_observation(self) -> np.ndarray:
        r"""Uncertainty about the inverse along the current observation.

        Computes the matrix-vector product :math:`W^H_{i-1}y_i` between the covariance
        factor of the inverse model and the current observation.
        """
        return self.belief.Ainv.cov.A @ self.observation.obsA

    @cached_property
    def sqnorm_covfactorA_action(self) -> float:
        r"""Squared norm :math:`\lVert W^A_{i-1}s_i\rVert^2` of the matrix covariance
        factor applied to the current action."""
        return (self.covfactorA_action.T @ self.covfactorA_action).item()

    @cached_property
    def sqnorm_covfactorH_observation(self) -> float:
        r"""Squared norm :math:`\lVert W^H_{i-1}y_i\rVert^2` of the inverse covariance
        factor applied to the current observation."""
        return (self.covfactorH_observation.T @ self.covfactorH_observation).item()

    @cached_property
    def action_covfactorA_action(self) -> float:
        r"""Inner product :math:`s_i^\top W^A_{i-1} s_i` of the current action
        with respect to the covariance factor :math:`W^A_{i-1}` of the matrix model."""
        return self.action.actA.T @ self.covfactorA_action

    @cached_property
    def observation_covfactorH_observation(self) -> float:
        r"""Inner product :math:`y_i^\top W^H_{i-1} y_i` of the current observation
        with respect to the covariance factor :math:`W^H_{i-1}` of the inverse model."""
        return self.observation.obsA.T @ self.covfactorH_observation

    @cached_property
    def deltaA_invcovfactorA_deltaA(self) -> float:
        r"""Inner product :math:`\Delta_i (W^A_{i-1})^{-1} \Delta_i` of the residual
        with respect to the inverse of the covariance factor."""
        return (
            self.deltaA.T @ scipy.sparse.linalg.cg(self.belief.A.cov.A, self.deltaA)[0]
        ).item()

    @cached_property
    def sum_deltaA_invgramA_deltaA(self) -> float:
        r"""Sum of inner products :math:`\Delta^A_i G^A_{i-1}^{-1} \Delta^A_i` of the
        matrix residual and the inverse Gram matrix."""
        if self.prev_cache is None:
            prev_sum_delta_invgram_delta = 0.0
        else:
            prev_sum_delta_invgram_delta = self.prev_cache.sum_deltaA_invgramA_deltaA
        return prev_sum_delta_invgram_delta + (
            2 * self.deltaA_invcovfactorA_deltaA / self.action_covfactorA_action
            - (self.deltaA_action / self.action_covfactorA_action) ** 2
        )

    @cached_property
    def meanA_update_op(self) -> linops.LinearOperator:
        """Rank 2 update term for the mean of the matrix model."""
        u = self.covfactorA_action / self.action_covfactorA_action
        v = self.deltaA - 0.5 * self.deltaA_action * u

        return linops.LinearOperator(
            shape=self.belief.A.shape,
            matvec=lambda x: u * (v.T @ x) + v * (u.T @ x),
            matmat=lambda x: u @ (v.T @ x) + v @ (u.T @ x),
        )

    @cached_property
    def meanH_update_op(self) -> linops.LinearOperator:
        """Rank 2 update term for the mean of the inverse model."""
        u = self.covfactorH_observation / self.observation_covfactorH_observation
        v = self.deltaH - 0.5 * self.deltaH_observation * u

        return linops.LinearOperator(
            shape=self.belief.Ainv.shape,
            matvec=lambda x: u * (v.T @ x) + v * (u.T @ x),
            matmat=lambda x: u @ (v.T @ x) + v @ (u.T @ x),
        )

    @cached_property
    def covfactorA_update_ops(self) -> Tuple[linops.LinearOperator, ...]:
        """Rank 1 downdate term(s) for the covariance factors of the matrix model."""
        u = self.covfactorA_action / self.action_covfactorA_action

        covfactor_update_op = linops.LinearOperator(
            shape=self.belief.A.shape,
            matvec=lambda x: self.covfactorA_action * (u.T @ x),
            matmat=lambda x: self.covfactorA_action @ (u.T @ x),
        )
        return covfactor_update_op, covfactor_update_op

    @cached_property
    def covfactorH_update_ops(self) -> Tuple[linops.LinearOperator, ...]:
        """Rank 1 downdate term(s) for the covariance factors of the inverse model."""
        u = self.covfactorH_observation / self.observation_covfactorH_observation

        covfactor_update_op = linops.LinearOperator(
            shape=self.belief.Ainv.shape,
            matvec=lambda x: self.covfactorH_observation * (u.T @ x),
            matmat=lambda x: self.covfactorH_observation @ (u.T @ x),
        )
        return covfactor_update_op, covfactor_update_op

    @cached_property
    def meanA_update_batch(self) -> linops.LinearOperator:
        """Matrix model mean update term for all actions and observations."""
        if self.prev_cache.data is None:
            return self.meanA_update_op
        return self.prev_cache.meanA_update_batch + self.meanA_update_op

    @cached_property
    def meanH_update_batch(self) -> linops.LinearOperator:
        """Inverse model mean update term for all actions and observations."""
        if self.prev_cache.data is None:
            return self.meanH_update_op
        return self.prev_cache.meanH_update_batch + self.meanH_update_op

    @cached_property
    def covfactorA_update_batch(self) -> Tuple[linops.LinearOperator, ...]:
        """Matrix model covariance factor downdate term for all actions and
        observations."""
        if self.prev_cache.data is None:
            return self.covfactorA_update_ops
        return tuple(
            map(
                lambda x, y: x + y,
                self.prev_cache.covfactorA_update_batch,
                self.covfactorA_update_ops,
            )
        )

    @cached_property
    def covfactorH_update_batch(self) -> Tuple[linops.LinearOperator, ...]:
        """Inverse model covariance factor downdate term for all actions and
        observations."""
        if self.prev_cache.data is None:
            return self.covfactorH_update_ops
        return tuple(
            map(
                lambda x, y: x + y,
                self.prev_cache.covfactorH_update_batch,
                self.covfactorH_update_ops,
            )
        )


class _SystemMatrixSymmetricNormalLinearObsBeliefUpdate(LinearSolverQoIBeliefUpdate):
    """Belief update for the system matrix assuming symmetrix matrix-variate normal
    priors and linear observations."""

    def __call__(
        self,
        problem: LinearSystem,
        hyperparams: "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams",
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> rvs.Normal:
        """Updated belief for the matrix."""
        if hyperparams is None or not isinstance(hyperparams, LinearSystemNoise):
            mean = (
                linops.aslinop(self.prior.A.mean)
                + solver_state.cache.meanA_update_batch
            )
            cov = linops.SymmetricKronecker(
                self.prior.A.cov.A - solver_state.cache.covfactorA_update_batch[0]
            )
        elif isinstance(hyperparams.A_eps.cov.A, linops.ScalarMult):
            eps_sq = hyperparams.A_eps.cov.A.scalar
            mean = (
                linops.aslinop(self.prior.A.mean)
                + 1 / (1 + eps_sq) * solver_state.cache.meanA_update_batch
            )

            cov = linops.SymmetricKronecker(
                self.prior.A.cov.A
                - 1 / (1 + eps_sq) * solver_state.cache.covfactorA_update_batch[0]
            ) + linops.SymmetricKronecker(
                eps_sq / (1 + eps_sq) * solver_state.cache.covfactorA_update_batch[1]
            )
        else:
            raise NotImplementedError(
                "Belief updated for general noise not implemented."
            )
        return rvs.Normal(mean=mean, cov=cov)


class _InverseMatrixSymmetricNormalLinearObsBeliefUpdate(LinearSolverQoIBeliefUpdate):
    """Belief update for the inverse assuming symmetrix matrix-variate normal priors and
    linear observations."""

    def __call__(
        self,
        problem: LinearSystem,
        hyperparams: "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams",
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> rvs.Normal:
        """Updated belief for the inverse matrix."""
        if hyperparams is None or not isinstance(hyperparams, LinearSystemNoise):
            mean = (
                linops.aslinop(self.prior.Ainv.mean)
                + solver_state.cache.meanH_update_batch
            )
            cov = linops.SymmetricKronecker(
                self.prior.Ainv.cov.A - solver_state.cache.covfactorH_update_batch[0]
            )
        elif isinstance(hyperparams.A_eps.cov.A, linops.ScalarMult):
            eps_sq = hyperparams.A_eps.cov.A.scalar
            mean = (
                linops.aslinop(self.prior.Ainv.mean)
                + 1 / (1 + eps_sq) * solver_state.cache.meanH_update_batch
            )

            cov = linops.SymmetricKronecker(
                self.prior.Ainv.cov.A
                - 1 / (1 + eps_sq) * solver_state.cache.covfactorH_update_batch[0]
            ) + linops.SymmetricKronecker(
                eps_sq / (1 + eps_sq) * solver_state.cache.covfactorH_update_batch[1]
            )
        else:
            raise NotImplementedError(
                "Belief updated for general noise not implemented."
            )
        return rvs.Normal(mean=mean, cov=cov)


class _SolutionSymmetricNormalLinearObsBeliefUpdate(LinearSolverQoIBeliefUpdate):
    """Belief update for the solution assuming symmetric matrix-variate normal priors
    and linear observations."""

    def __call__(
        self,
        problem: LinearSystem,
        hyperparams: "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams",
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> Optional[Union[rvs.Normal, np.ndarray]]:
        """Updated belief about the solution."""
        return (
            solver_state.cache.belief.x.mean
            + solver_state.cache.step_size * solver_state.cache.action.actA
        )


class _RightHandSideSymmetricNormalLinearObsBeliefUpdate(LinearSolverQoIBeliefUpdate):
    """Belief update for the right hand side assuming symmetric matrix-variate normal
    priors and linear observations."""

    def __call__(
        self,
        problem: LinearSystem,
        hyperparams: "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams",
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> Union[rvs.Constant, rvs.Normal]:
        """Updated belief for the right hand side."""
        if hyperparams is None or not isinstance(hyperparams, LinearSystemNoise):
            return rvs.asrandvar(problem.b)
        elif hyperparams.b_eps is None:
            return rvs.asrandvar(problem.b)
        else:
            # TODO replace this with Gaussian inference
            return (
                solver_state.belief.b * solver_state.info.iteration
                + solver_state.cache.observation.obsb
            ) / (solver_state.info.iteration + 1)


class SymmetricNormalLinearObsBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Belief update for a symmetric matrix-variate Normal belief and linear
    observations.

    Updates the posterior beliefs over the quantities of interest of the linear system
    under symmetric matrix-variate Gaussian prior(s) on :math:`A` and / or :math:`H`.
    Observations are assumed to be linear.

    Parameters
    ----------
    """

    def __init__(
        self,
        prior: SymmetricNormalLinearSystemBelief,
        cache_type=_SymmetricNormalLinearObsCache,
        x_belief_update_type=_SolutionSymmetricNormalLinearObsBeliefUpdate,
        A_belief_update_type=_SystemMatrixSymmetricNormalLinearObsBeliefUpdate,
        Ainv_belief_update_type=_InverseMatrixSymmetricNormalLinearObsBeliefUpdate,
        b_belief_update_type=_RightHandSideSymmetricNormalLinearObsBeliefUpdate,
    ):
        super().__init__(
            prior=prior,
            cache_type=cache_type,
            x_belief_update_type=x_belief_update_type,
            A_belief_update_type=A_belief_update_type,
            Ainv_belief_update_type=Ainv_belief_update_type,
            b_belief_update_type=b_belief_update_type,
        )
