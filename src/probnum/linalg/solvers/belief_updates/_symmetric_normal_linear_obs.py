try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import dataclasses
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers import belief_updates
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.hyperparams import (
    LinearSolverHyperparams,
    LinearSystemNoise,
)
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["SymmetricNormalLinearObsBeliefUpdate"]


@dataclasses.dataclass
class _SolutionSymmetricNormalLinearObsBeliefUpdateState(
    belief_updates.LinearSolverBeliefUpdateState
):
    r"""Quantities used in the belief update of the solution."""

    def __init__(
        self,
        problem: LinearSystem,
        prior: LinearSystemBelief,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[LinearSolverHyperparams] = None,
        prev_state: Optional["_MatrixSymmetricNormalLinearObsBeliefUpdateState"] = None,
    ):
        super().__init__(
            problem=problem,
            prior=prior,
            belief=belief,
            action=action,
            observation=observation,
            hyperparams=hyperparams,
            prev_state=prev_state,
        )

    @cached_property
    def action_observation(self) -> float:
        """Inner product :math:`s_i^\top y_j` between the current action and
        observation."""
        return (self.action.T @ self.observation).item()

    @cached_property
    def log_rayleigh_quotients(self) -> List[float]:
        r"""Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(
        s_i^\top s_i)`."""
        return self.prev_state.log_rayleigh_quotients + [
            np.log(self.action_observation)
            - np.log((self.action.T @ self.action)).item()
        ]

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r = A x_i- b` of the solution estimate
        :math:`x_i=\mathbb{E}[\mathsf{x}]` at iteration :math:`i`."""
        if self.hyperparams is not None or self._prev_state is None:
            return self.problem.A @ self.belief.x.mean - self.problem.b
        else:
            return self.prev_state.residual + self.step_size * self.observation

    @cached_property
    def step_size(self) -> np.ndarray:
        r"""Step size :math:`\alpha_i` of the solver viewed as a quadratic optimizer
        taking steps :math:`x_{i+1} = x_i + \alpha_i s_i`."""
        if self.hyperparams is None:
            return (
                -self.action.T @ self.prev_state.residual / self.action_observation
            ).item()
        else:
            raise NotImplementedError

    def updated_belief(
        self, hyperparams: Optional[LinearSystemNoise] = None
    ) -> Optional[Union[rvs.Normal, np.ndarray]]:
        """Updated belief about the solution."""
        if hyperparams is None and self.prev_state is not None:
            return self.prev_state.residual + self.step_size * self.observation
        else:
            # Belief is induced from inverse and rhs
            return None


@dataclasses.dataclass
class _MatrixSymmetricNormalLinearObsBeliefUpdateState(
    belief_updates.LinearSolverBeliefUpdateState
):
    r"""Quantities used in the belief update of a matrix quantity of interest.

    Collects the belief update terms for a quantity of interest of a linear
    system, i.e. additive terms for the mean and covariance (factors).
    """

    def __init__(
        self,
        qoi: str,
        problem: LinearSystem,
        prior: LinearSystemBelief,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[LinearSolverHyperparams] = None,
        prev_state: Optional["_MatrixSymmetricNormalLinearObsBeliefUpdateState"] = None,
    ):
        if qoi == "A":
            self.qoi_prior = prior.A
            self.qoi_belief = belief.A
        elif qoi == "Ainv":
            self.qoi_prior = prior.Ainv
            self.qoi_belief = belief.Ainv
        else:
            raise ValueError("Unknown matrix quantity of interest.")

        super().__init__(
            problem=problem,
            prior=prior,
            belief=belief,
            action=action,
            observation=observation,
            hyperparams=hyperparams,
            prev_state=prev_state,
        )

    @cached_property
    def delta(self) -> np.ndarray:
        r"""Residual :math:`\Delta` between observation and prediction."""
        return self.observation - self.qoi_belief.mean @ self.action

    @cached_property
    def delta_action(self) -> float:
        r"""Inner product :math:`\Delta^\top s` between residual and action."""
        return self.delta.T @ self.action

    @cached_property
    def covfactor_action(self) -> np.ndarray:
        r"""Uncertainty about the matrix along the current action.

        Computes the matrix-vector product :math:`W_{i-1}s_i` between the covariance
        factor of the matrix model and the current action.
        """
        return self.qoi_belief.cov.A @ self.action

    @cached_property
    def sqnorm_covfactor_action(self) -> float:
        r"""Squared norm :math:`\lVert W_{i-1}s_i\rVert^2` of the covariance factor
        applied to the action."""
        return (self.covfactor_action.T @ self.covfactor_action).item()

    @cached_property
    def action_covfactor_action(self) -> float:
        r"""Inner product :math:`s_i^\top W_{i-1} s_i` of the current action
        with respect to the covariance factor :math:`W_{i-1}` of the matrix model."""
        return self.action.T @ self.covfactor_action

    @cached_property
    def delta_invcovfactor_delta(self) -> float:
        r"""Inner product :math:`\Delta_i W_{i-1}^{-1} \Delta_i` of the residual and
        the inverse covariance factor."""
        return self.delta.T @ np.linalg.solve(self.qoi_belief.cov.A, self.delta)

    @cached_property
    def sum_delta_invgram_delta(self) -> float:
        r"""Sum of inner products :math:`\Delta_i G_{i-1}^{-1} \Delta_i` of the
        residual and the inverse Gram matrix."""
        return self.prev_state.sum_delta_invgram_delta + (
            2 * self.delta_invcovfactor_delta / self.action_covfactor_action
            - (self.delta_action / self.action_covfactor_action) ** 2
        )

    @cached_property
    def mean_update_op(self) -> linops.LinearOperator:
        r"""Rank 2 update term for the mean of the matrix model."""
        u = self.covfactor_action / self.action_covfactor_action
        v = self.delta - 0.5 * self.delta_action * u

        def mv(x):
            return u * (v.T @ x) + v * (u.T @ x)

        def mm(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(
            shape=(self.action.shape[0], self.action.shape[0]), matvec=mv, matmat=mm
        )

    @cached_property
    def covfactor_update_ops(self) -> Tuple[linops.LinearOperator, ...]:
        r"""Rank 1 downdate term(s) for the covariance factors of the matrix model."""
        u = self.covfactor_action / self.action_covfactor_action

        def mv(x):
            return self.covfactor_action * (u.T @ x)

        def mm(x):
            return self.covfactor_action @ (u.T @ x)

        # Covariance factor update linop WSU'
        covfactor_update_op = linops.LinearOperator(
            shape=(self.action.shape[0], self.action.shape[0]),
            matvec=mv,
            matmat=mm,
        )
        return covfactor_update_op, covfactor_update_op

    @cached_property
    def mean_update_batch(self) -> linops.LinearOperator:
        """Mean update term for all actions and observations."""
        if self.prev_state is None:
            return self.mean_update_op
        return self.prev_state.mean_update_batch + self.mean_update_op

    @cached_property
    def covfactor_updates_batch(self) -> Tuple[linops.LinearOperator, ...]:
        """Covariance factor downdate term for all actions and observations."""
        if self.prev_state is None:
            return self.covfactor_update_ops
        return tuple(
            map(
                lambda x, y: x + y,
                self.prev_state.covfactor_updates_batch,
                self.covfactor_update_ops,
            )
        )

    def updated_belief(self, hyperparams: LinearSystemNoise = None) -> rvs.Normal:
        """Updated belief for the matrix model."""
        if hyperparams is None:
            mean = linops.aslinop(self.qoi_prior.mean) + self.mean_update_batch
            cov = linops.SymmetricKronecker(
                self.qoi_prior.cov.A - self.covfactor_updates_batch[0]
            )
        elif isinstance(hyperparams.A_eps, linops.ScalarMult):
            eps_sq = hyperparams.A_eps.cov.A.scalar
            mean = linops.aslinop(self.qoi_prior.mean) + self.mean_update_batch / (
                1 + eps_sq
            )

            cov = linops.SymmetricKronecker(
                self.qoi_prior.cov.A - self.covfactor_updates_batch[0] / (1 + eps_sq)
            ) + linops.SymmetricKronecker(
                eps_sq / (1 + eps_sq) * self.covfactor_updates_batch[1]
            )
        else:
            raise NotImplementedError(
                "Belief updated for general noise not implemented."
            )
        return rvs.Normal(mean=mean, cov=cov)


@dataclasses.dataclass
class _RightHandSideSymmetricNormalLinearObsBeliefUpdateState(
    belief_updates.LinearSolverBeliefUpdateState
):
    r"""Quantities used in the belief update of the right hand side."""

    def __init__(
        self,
        problem: LinearSystem,
        prior: LinearSystemBelief,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[LinearSolverHyperparams] = None,
        prev_state: Optional[
            "_RightHandSideSymmetricNormalLinearObsBeliefUpdateState"
        ] = None,
    ):
        super().__init__(
            problem=problem,
            prior=prior,
            belief=belief,
            action=action,
            observation=observation,
            hyperparams=hyperparams,
            prev_state=prev_state,
        )

    def updated_belief(
        self, hyperparams: LinearSystemNoise = None
    ) -> Union[rvs.Constant, rvs.Normal]:
        """Updated belief for the right hand side."""
        if hyperparams is None:
            return rvs.asrandvar(self.problem.b)
        else:
            raise NotImplementedError


class SymmetricNormalLinearObsBeliefUpdate(belief_updates.LinearSolverBeliefUpdate):
    r"""Belief update for a symmetric matrix-variate Normal belief and linear
    observations.

    Updates the posterior beliefs over the quantities of interest of the linear system
    under symmetric matrix-variate Gaussian prior(s) on :math:`A` and / or :math:`H`.
    Observations are assumed to be linear.

    Parameters
    ----------
    x_belief_update_state_type :
    A_belief_update_state_type :
    Ainv_belief_update_state_type :
    b_belief_update_state_type :
    """

    def __init__(
        self,
        x_belief_update_state_type=_SolutionSymmetricNormalLinearObsBeliefUpdateState,
        A_belief_update_state_type=partial(
            _MatrixSymmetricNormalLinearObsBeliefUpdateState, qoi="A"
        ),
        Ainv_belief_update_state_type=partial(
            _MatrixSymmetricNormalLinearObsBeliefUpdateState, qoi="Ainv"
        ),
        b_belief_update_state_type=_RightHandSideSymmetricNormalLinearObsBeliefUpdateState,
    ):
        super().__init__(
            x_belief_update_state_type=x_belief_update_state_type,
            A_belief_update_state_type=A_belief_update_state_type,
            Ainv_belief_update_state_type=Ainv_belief_update_state_type,
            b_belief_update_state_type=b_belief_update_state_type,
        )
