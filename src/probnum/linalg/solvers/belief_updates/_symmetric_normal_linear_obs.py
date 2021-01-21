try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import dataclasses
from typing import List, Optional, Tuple, Union

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers import belief_updates
from probnum.linalg.solvers._state import (
    LinearSolverData,
    LinearSolverMiscQuantities,
    LinearSolverState,
)
from probnum.linalg.solvers.beliefs import LinearSystemBelief, LinearSystemNoise
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
        qoi_prior: rvs.Normal,
        qoi_belief: rvs.Normal,
        action: np.ndarray,
        observation: np.ndarray,
        noise: LinearSystemNoise,
        prev_state: Optional[
            "_SolutionSymmetricNormalLinearObsBeliefUpdateState"
        ] = None,
    ):
        self.noise = noise
        super().__init__(
            problem=problem,
            qoi_prior=qoi_prior,
            qoi_belief=qoi_belief,
            action=action,
            observation=observation,
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
        if self.noise is None and self.prev_state is not None:
            return self.prev_state.residual + self.step_size * self.observation
        else:
            return self.problem.A @ self.qoi_belief.mean - self.problem.b

    @cached_property
    def step_size(self) -> np.ndarray:
        r"""Step size :math:`\alpha_i` of the solver viewed as a quadratic optimizer
        taking steps :math:`x_{i+1} = x_i + \alpha_i s_i`."""
        return (
            -self.action.T @ self.prev_state.residual / self.action_observation
        ).item()

    def updated_belief(self) -> np.ndarray:
        """Updated belief about the solution."""
        if self.noise is None and self.prev_state is not None:
            mean = self.prev_state.residual + self.step_size * self.observation
            raise NotImplementedError
        else:
            raise NotImplementedError


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
        problem: LinearSystem,
        qoi_prior: rvs.Normal,
        qoi_belief: rvs.Normal,
        action: np.ndarray,
        observation: np.ndarray,
        noise: LinearSystemNoise,
        prev_state: Optional["_MatrixSymmetricNormalLinearObsBeliefUpdateState"] = None,
    ):
        self.noise = noise
        super().__init__(
            problem=problem,
            qoi_prior=qoi_prior,
            qoi_belief=qoi_belief,
            action=action,
            observation=observation,
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

        mean_update_op = linops.LinearOperator(
            shape=(self.action.shape[0], self.action.shape[0]), matvec=mv, matmat=mm
        )

        if self.noise is None:
            return mean_update_op
        else:
            eps_sq = self.noise.A_eps.cov.A.scalar
            return 1 / (1 + eps_sq) * mean_update_op

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

        if self.noise is None:
            return (covfactor_update_op,)
        else:
            eps_sq = self.noise.A_eps.cov.A.scalar
            return (
                covfactor_update_op / (1 + eps_sq),
                eps_sq / (1 + eps_sq) * covfactor_update_op,
            )

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

    @cached_property
    def updated_belief(self) -> rvs.Normal:
        """Updated belief for the matrix model."""
        mean = self.qoi_belief.mean + self.mean_update_batch
        if self.noise is None:
            cov = self.qoi_belief.cov - linops.SymmetricKronecker(
                self.covfactor_updates_batch[0]
            )
        else:
            cov = linops.SymmetricKronecker(
                self.qoi_belief.cov.A - self.covfactor_updates_batch[0]
            ) + linops.SymmetricKronecker(self.covfactor_updates_batch[1])
        return rvs.Normal(mean=mean, cov=cov)


@dataclasses.dataclass
class _RightHandSideSymmetricNormalLinearObsBeliefUpdateState:
    r"""Quantities used in the belief update of the right hand side."""

    def __init__(
        self,
        problem: LinearSystem,
        qoi_prior: rvs.Normal,
        qoi_belief: rvs.Normal,
        action: np.ndarray,
        observation: np.ndarray,
        noise: LinearSystemNoise,
        prev_state: Optional[
            "_RightHandSideSymmetricNormalLinearObsBeliefUpdateState"
        ] = None,
    ):
        self.noise = noise
        super().__init__(
            problem=problem,
            qoi_prior=qoi_prior,
            qoi_belief=qoi_belief,
            action=action,
            observation=observation,
            prev_state=prev_state,
        )

    @cached_property
    def updated_belief(self) -> Union[rvs.Constant, rvs.Normal]:
        """Updated belief for the right hand side."""
        if self.noise is None:
            return rvs.asrandvar(self.problem.b)
        else:
            raise NotImplementedError


class SymmetricNormalLinearObsBeliefUpdate(belief_updates.LinearSolverBeliefUpdate):
    r"""Belief update for a symmetric matrix-variate Normal belief and linear
    observations.

    Updates the posterior beliefs over the quantities of interest of the linear system
    under symmetric matrix-variate Gaussian prior(s) on :math:`A` and / or :math:`H`.
    Observations are assumed to be linear.

    Examples
    --------

    """

    def update_solver_state(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional[LinearSolverState],
    ) -> LinearSolverState:
        """Perform a lazy update of the linear solver state."""
        if solver_state is None:
            # TODO init solver state
            pass
        else:
            data = LinearSolverData(
                actions=solver_state.data.actions + [action],
                observations=solver_state.data.observations + [observation],
            )
            state_x = _SolutionSymmetricNormalLinearObsBeliefUpdateState(
                problem=solver_state.problem,
                qoi_prior=solver_state.prior.x,
                qoi_belief=solver_state.belief.x,
                action=action,
                observation=observation,
                noise=None,  # TODO
                prev_state=solver_state.misc.x,
            )

            return LinearSolverState(
                problem=solver_state.problem,
                prior=solver_state.prior,
                belief=solver_state.belief,
                data=data,
                info=solver_state.info,
                misc=LinearSolverMiscQuantities(
                    x=state_x,
                    A=state_A,
                    Ainv=state_Ainv,
                    b=state_b,
                ),
            )

    def update_belief(self, solver_state: Optional[LinearSolverState]):
        if solver_state is None:
            # TODO init solver state
            pass
        else:
            return LinearSystemBelief(
                x=solver_state.misc.x.updated_belief,
                Ainv=solver_state.misc.Ainv.updated_belief,
                A=solver_state.misc.A.updated_belief,
                b=solver_state.misc.b.updated_belief,
            )
