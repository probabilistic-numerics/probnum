"""State of a probabilistic linear solver."""

import dataclasses
from typing import Optional

from probnum.linalg.solvers.data import LinearSolverData

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


import numpy as np

import probnum
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverInfo",
    "LinearSolverCache",
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
    stopping_criterion: Optional[
        "probnum.linalg.solvers.stop_criteria.StoppingCriterion"
    ] = None


@dataclasses.dataclass
class LinearSolverCache:
    r"""Miscellaneous cached quantities.

    Used to efficiently select an action, optimize hyperparameters and to update the
    belief. This class is intended to be subclassed to store any quantities which are
    reused multiple times within the linear solver and thus can be cached within the
    current iteration.

    Parameters
    ----------
    problem
        Linear system to be solved.
    belief
        (Updated) belief over the quantities of interest of the linear system.
    hyperparams
    action
    observation
    prev_cache
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams"
        ] = None,
        action: Optional["probnum.linalg.solvers.LinearSolverAction"] = None,
        observation: Optional["probnum.linalg.solvers.LinearSolverObservation"] = None,
        prev_cache: Optional["LinearSolverCache"] = None,
    ):
        # pylint: disable="too-many-arguments"

        self.problem = problem
        self.belief = belief
        self.hyperparams = hyperparams
        self.action = action
        self.observation = observation
        self.prev_cache = prev_cache

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r = A x_i- b` of the solution estimate
        :math:`x_i=\mathbb{E}[\mathsf{x}]` at iteration :math:`i`."""
        return self.problem.A @ self.belief.x.mean - self.problem.b

    @classmethod
    def from_new_data(
        cls,
        action: Optional["probnum.linalg.solvers.LinearSolverAction"],
        observation: Optional["probnum.linalg.solvers.LinearSolverObservation"],
        prev_cache: "LinearSolverCache",
    ):
        """Create new cached quantities from new data."""
        return cls(
            problem=prev_cache.problem,
            belief=prev_cache.belief,
            action=action,
            observation=observation,
            prev_cache=prev_cache,
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
        Information about the convergence of the linear solver.
    problem
        Linear system to be solved.
    prior
        Prior belief over the quantities of interest of the linear system.
    data
        Performed actions and collected observations of the linear system.
    belief
        (Updated) belief over the quantities of interest of the linear system.
    misc
        Miscellaneous (cached) quantities to efficiently select an action,
        optimize hyperparameters and to update the belief.
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        prior: Optional["probnum.linalg.solvers.beliefs.LinearSystemBelief"] = None,
        data: Optional[LinearSolverData] = None,
        info: Optional[LinearSolverInfo] = None,
        misc: Optional[LinearSolverCache] = None,
    ):
        # pylint: disable="too-many-arguments"

        self.problem = problem
        self.belief = belief
        self.data = (
            data if data is not None else LinearSolverData(actions=[], observations=[])
        )
        self.prior = prior if prior is not None else belief
        self.info = info if info is not None else LinearSolverInfo()
        self.misc = (
            misc
            if misc is not None
            else LinearSolverCache(problem=problem, belief=belief)
        )

    @classmethod
    def from_new_data(
        cls,
        action: np.ndarray,
        observation: np.ndarray,
        prev_state: "LinearSolverState",
    ):
        """Create a new solver state from a previous one and newly observed data.

        Parameters
        ----------
        action :
            Action taken by the solver given by its policy.
        observation :
            Observation of the linear system for the corresponding action.
        prev_state :
            Previous linear solver state prior to observing new data.
        """
        data = LinearSolverData(
            actions=prev_state.data.actions + [action],
            observations=prev_state.data.observations + [observation],
        )
        misc = LinearSolverCache.from_new_data(
            action=action, observation=observation, prev_cache=prev_state.misc
        )

        return cls(
            problem=prev_state.problem,
            prior=prev_state.prior,
            data=data,
            belief=prev_state.belief,
            info=prev_state.info,
            misc=misc,
        )

    @classmethod
    def from_updated_belief(
        cls,
        updated_belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        prev_state: "LinearSolverState",
    ):
        """Create a new solver state from an updated belief.

        Parameters
        ----------
        updated_belief :
            Updated belief over the quantities of interest after observing data.
        prev_state :
            Previous linear solver state updated with new data, but prior to the
            belief update.
        """

        return cls(
            problem=prev_state.problem,
            prior=prev_state.prior,
            data=prev_state.data,
            belief=updated_belief,
            info=prev_state.info,
            misc=prev_state.misc,
        )
