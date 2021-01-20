import dataclasses
from typing import Callable, List, Optional

import numpy as np

from probnum.linalg.solvers import belief_updates, beliefs, stop_criteria

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from functools import lru_cache

from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverInfo",
    "LinearSolverData",
    "LinearSolverMiscQuantities",
    "LinearSolverState",
]


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
    r"""Data about a numerical problem.

    Actions and observations collected by a probabilistic linear solver via
    its observation process.

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
    belief. This class is intended to be subclassed to store any quantities which are
    reused multiple times within the linear solver and thus can be cached within the
    current iteration.

    Parameters
    ----------
    iteration
        Current iteration of the solver.
    problem
        Linear system to be solved.
    belief
        Belief over the quantities of the linear system.
    data
        Performed actions and collected observations of the linear system.
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
        iteration: int,
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
        # TODO belief update states for x, A, Ainv, b here?
    ):
        self._iteration = iteration
        self._problem = problem
        self._belief = belief
        self._data = data
        self._residual = residual
        self._step_sizes = step_sizes
        self._action_obs_innerprods = action_obs_innerprods
        self._log_rayleigh_quotients = log_rayleigh_quotients

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r_i = Ax_i - b` of the solution estimate."""
        if self._residual is None:
            return self._problem.A @ self._belief.x.mean - self._problem.b
        else:
            self._residual()

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
                    "nk,nk->k", self._data.actions_arr, self._data.observations_arr
                )
            return self._data.actions_arr.T @ self._data.observations_arr
        else:
            return self._action_obs_innerprods(self._data, diag)

    @cached_property
    def log_rayleigh_quotients(self) -> np.ndarray:
        r"""Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(
        s_i^\top s_i)`."""
        if self._log_rayleigh_quotients is None:
            return np.log(self.action_obs_innerprods(diag=True)) - np.log(
                np.einsum("nk,nk->k", self._data.actions_arr, self._data.actions_arr)
            )
        else:
            return self._log_rayleigh_quotients(self._problem, self._data)

    @cached_property
    def step_sizes(self) -> np.ndarray:
        r"""Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer
        taking steps :math:`x_{i+1} = x_i + \alpha_i s_i`."""
        if self._step_sizes is None:
            raise NotImplementedError
        else:
            return self._step_sizes(self._problem, self._belief, self._data)

    # Noisy solver

    @cached_property
    def sq_resid_norm_gram(self) -> float:
        r"""Squared norm of the residual :math:`\Delta_i G_{i-1} \Delta_i` with
        respect to the Gramian :math:`G_{i-1}`."""
        if self._sq_resid_norm_gram is None:
            delta_covfactorA_delta = self.residA.T @ self._belief.A.cov.A @ self.residA
            return (
                2 * delta_covfactorA_delta / self.action_covfactorA_action
                - (self.residA_action / self.action_covfactorA_action) ** 2
            )


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
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: beliefs.LinearSystemBelief,
        data: LinearSolverData,
        info: Optional[LinearSolverInfo] = None,
        misc: Optional[LinearSolverMiscQuantities] = None,
    ):

        self.problem = problem
        self.belief = belief
        self.data = data
        if info is None:
            self.info = LinearSolverInfo()
        else:
            self.info = info
        if misc is None:
            self.misc = LinearSolverMiscQuantities(
                iteration=self.info.iteration,
                problem=problem,
                belief=belief,
                data=data,
            )
        else:
            self.misc = misc
