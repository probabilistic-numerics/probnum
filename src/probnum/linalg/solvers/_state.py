import dataclasses
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from probnum import PNMethodInfo, PNMethodState
from probnum.linalg.solvers import belief_updates, beliefs, stop_criteria

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from functools import lru_cache

from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSolverInfo", "LinearSolverState"]


@dataclasses.dataclass
class LinearSolverInfo:
    """Information on the solve by the probabilistic numerical method.

    Parameters
    ----------
    iteration
        Current iteration :math:`i` of the solver.
    has_converged
        Has the solver converged?
    stopping_criterion
        Stopping criterion which caused termination of the solver.
    """

    iteration: int = 0
    has_converged: bool = False
    stopping_criterion: Optional[List[stop_criteria.StoppingCriterion]] = None


@dataclasses.dataclass
class LinearSolverData:
    r"""Data collected by a probabilistic linear solver via its observation process.

    Parameters
    ----------
    actions
        Performed actions.
    observations
        Collected observations of the problem.
    """
    actions: List
    observations: List

    @cached_property
    def actions_arr(self) -> np.ndarray:
        """Array of performed actions."""
        return np.hstack(self.actions)

    @cached_property
    def observations_arr(self) -> np.ndarray:
        """Array of performed observations."""
        return np.hstack(self.observations)


@dataclasses.dataclass
class LinearSolverMiscQuantities:
    r"""Miscellaneous (cached) quantities.

    Used to efficiently select an action, optimize hyperparameters and to update the
    belief.

    This class is intended to be subclassed to store any quantities which are reused
    multiple times within the linear solver and thus can be cached within the current
    iteration.
    """


@dataclasses.dataclass
class LinearSolverState:
    r"""State of a probabilistic linear solver.

    The solver state contains miscellaneous quantities computed during an iteration
    of a probabilistic linear solver. The solver state is passed between the
    different components of the solver and may be used by them.

    For example the residual :math:`r_i = Ax_i - b` can (depending on the prior) be
    updated more efficiently than in :math:`\mathcal{O}(n^2)` and is therefore part
    of the solver state and passed to the stopping criteria.

    Parameters
    ----------

    info
        Information about the convergence of the linear solver
    problem
        Linear system to be solved.
    belief
        Belief over the quantities of the linear system.
    data
        Performed actions and collected observations of the linear system.
    misc
        Miscellaneous (cached) quantities to efficiently select an action,
        optimize hyperparameters and to update the belief.

    residual
        Residual :math:`r = A x_i- b` of the solution estimate
        :math:`x_i=\mathbb{E}[\mathsf{x}]` at iteration :math:`i`.
    step_sizes
        Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer taking
        steps :math:`x_{i+1} = x_i + \alpha_i s_i`.
    action_obs_innerprods
        Inner product(s) :math:`(S^\top Y)_{ij} = s_i^\top y_j` of actions
        and observations. If a vector, actions and observations are assumed to be
        conjugate, i.e. :math:`s_i^\top y_j =0` for :math:`i \neq j`.
    log_rayleigh_quotients
        Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(s_i^\top
        s_i)`.
    """

    def __init__(
        self,
        info: LinearSolverInfo,
        problem: LinearSystem,
        belief: beliefs.LinearSystemBelief,
        data: LinearSolverData,
        residual: Optional[
            Callable[
                [LinearSystem, beliefs.LinearSystemBelief, LinearSolverData], np.ndarray
            ]
        ] = None,
        step_sizes: Optional[
            Callable[
                [LinearSystem, beliefs.LinearSystemBelief, LinearSolverData], np.ndarray
            ]
        ] = None,
        action_obs_innerprods: Optional[
            Callable[[LinearSolverData, bool], np.ndarray]
        ] = None,
        log_rayleigh_quotients: Optional[
            Callable[[LinearSystem, LinearSolverData], np.ndarray]
        ] = None,
    ):
        self.info = info
        self.problem = problem
        self.belief = belief
        self.data = data
        self._residual = residual
        self._step_sizes = step_sizes
        self._action_obs_innerprods = action_obs_innerprods
        self._log_rayleigh_quotients = log_rayleigh_quotients

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r_i = Ax_i - b` of the solution estimate."""
        if self._residual is None:
            return self.problem.A @ self.belief.x.mean - self.problem.b
        else:
            self._residual(self.problem, self.belief, self.data)

    @lru_cache(maxsize=2)
    def action_obs_innerprods(self, diag=True) -> np.ndarray:
        r"""Inner products :math:`(S^\top Y)_{ij} = s_i^\top y_j` of actions
        and observations.

        Parameters
        ----------
        diag :
            Return only the diagonal :math:`s_i^\top y_i`.
        """
        if self._action_obs_innerprods is None:
            if diag:
                return np.einsum(
                    "nk,nk->k", self.data.actions_arr, self.data.observations_arr
                )
            return self.data.actions_arr.T @ self.data.observations_arr
        else:
            return self._action_obs_innerprods(self.data, diag)

    @cached_property
    def log_rayleigh_quotients(self) -> np.ndarray:
        r"""Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(
        s_i^\top s_i)`."""
        if self._log_rayleigh_quotients is None:
            return np.log(self.action_obs_innerprods(diag=True)) - np.log(
                np.einsum("nk,nk->k", self.data.actions_arr, self.data.actions_arr)
            )
        else:
            return self._log_rayleigh_quotients(self.problem, self.data)

    @cached_property
    def step_sizes(self) -> np.ndarray:
        r"""Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer
        taking steps :math:`x_{i+1} = x_i + \alpha_i s_i`."""
        if self._step_sizes is None:
            raise NotImplementedError
        else:
            return self._step_sizes(self.problem, self.belief, self.data)

    # Symmetric linear observations

    @cached_property
    def residA(self) -> np.ndarray:
        r"""Residual :math:`\Delta_i = y_i - A_{i-1}s_i` where :math:`A_{i-1}` is the
        expected value of the model for the system matrix."""
        if self._residA is None:
            return (
                self.data.observations[self.info.iteration]
                - self.belief.A.mean @ self.data.actions[self.info.iteration]
            )
        else:
            return self._residA(self.belief, self.data)

    @cached_property
    def residA_action(self) -> float:
        r"""Inner product :math:`\Delta_i^\top s_i` between residual and action."""
        if self._residA_action is None:
            return self.residA @ self.data.actions[self.info.iteration]
        else:
            return self._residA_action(self.belief, self.data)

    @cached_property
    def residAinv(self) -> np.ndarray:
        r"""Residual :math:`\Delta_i = s_i - H_{i-1}y_i` where :math:`H_{i-1}` is the
        expected value of the model for the inverse."""
        if self._residAinv is None:
            return (
                self.data.actions[self.info.iteration]
                - self.belief.Ainv.mean @ self.data.observations[self.info.iteration]
            )
        else:
            return self._residAinv(self.belief, self.data)

    @cached_property
    def covfactorA_action(self) -> np.ndarray:
        r"""Uncertainty about the matrix along the current action.

        Computes the matrix-vector product :math:`W_{i-1}s_i` between the covariance
        factor of the matrix model and the current action.
        """
        if self._covfactorA_action is None:
            return self.belief.A.cov.A @ self.data.actions[self.info.iteration]
        else:
            return self._covfactorA_action(self.belief, self.data)

    @cached_property
    def action_covfactorA_action(self) -> float:
        r"""Inner product :math:`s_i^\top W_{i-1} s_i` of the current action
        with respect to the covariance factor :math:`W_{i-1}` of the matrix model."""
        if self._action_covfactorA_action is None:
            return self.data.actions[self.info.iteration] @ self.covfactorA_action
        else:
            return self._action_covfactorA_action(self.belief, self.data)

    # Noisy solver

    @cached_property
    def sq_resid_norm_gram(self) -> float:
        r"""Squared norm of the residual :math:`\Delta_i G_{i-1} \Delta_i` with
        respect to the Gramian :math:`G_{i-1}`."""
        if self._sq_resid_norm_gram is None:
            delta_covfactorA_delta = self.residA.T @ self.belief.A.cov.A @ self.residA
            return (
                2 * delta_covfactorA_delta / self.action_covfactorA_action
                - (self.residA_action / self.action_covfactorA_action) ** 2
            )
