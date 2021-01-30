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
    prior
        Prior belief over the quantities of interest of the linear system.
    belief
        (Updated) belief over the quantities of interest of the linear system.
    hyperparams
        Hyperparameters of the linear solver.
    data
        Performed actions and collected observations of the linear system.
    prev_cache
        Cached quantities from the previous iteration.
    """

    def __init__(
        self,
        problem: LinearSystem,
        prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams"
        ] = None,
        data: Optional[LinearSolverData] = None,
        prev_cache: Optional["LinearSolverCache"] = None,
    ):
        # pylint: disable="too-many-arguments"

        self.problem = problem
        self.prior = prior
        self.belief = belief
        if hyperparams is None:
            self.hyperparams = belief.hyperparams
        else:
            self.hyperparams = hyperparams
        self.data = data
        self.prev_cache = prev_cache

    @property
    def action(self) -> Optional["probnum.linalg.solvers.LinearSolverAction"]:
        r"""Most recent action of the linear solver."""
        if self.data is None:
            return None
        return self.data.actions[-1]

    @property
    def observation(self) -> Optional["probnum.linalg.solvers.LinearSolverObservation"]:
        r"""Most recent action of the linear solver."""
        if self.data is None:
            return None
        return self.data.observations[-1]

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r = A x_i- b` of the solution estimate
        :math:`x_i=\mathbb{E}[\mathsf{x}]` at iteration :math:`i`."""
        return self.problem.A @ self.belief.x.mean - self.problem.b

    @classmethod
    def from_new_data(
        cls,
        action: "probnum.linalg.solvers.LinearSolverAction",
        observation: "probnum.linalg.solvers.LinearSolverObservation",
        prev_cache: "LinearSolverCache",
    ):
        """Create new cached quantities from new data."""
        if prev_cache.data is None:
            actions = [action]
            observations = [observation]
        else:
            actions = prev_cache.data.actions + [action]
            observations = prev_cache.data.observations + [observation]
        data = LinearSolverData(
            actions=actions,
            observations=observations,
        )
        return cls(
            problem=prev_cache.problem,
            prior=prev_cache.prior,
            belief=prev_cache.belief,
            data=data,
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
    cache
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
        cache: Optional[LinearSolverCache] = None,
    ):
        # pylint: disable="too-many-arguments"

        self.problem = problem
        self.belief = belief
        self.data = (
            data if data is not None else LinearSolverData(actions=[], observations=[])
        )
        self.prior = prior if prior is not None else belief
        self.info = (
            info
            if info is not None
            else LinearSolverInfo(iteration=len(self.data.actions))
        )
        self.cache = (
            cache
            if cache is not None
            else LinearSolverCache(
                problem=problem, prior=self.prior, belief=belief, data=self.data
            )
        )

    @classmethod
    def from_new_data(
        cls,
        action: "probnum.linalg.solvers.LinearSolverAction",
        observation: "probnum.linalg.solvers.LinearSolverObservation",
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
        cache = type(prev_state.cache).from_new_data(
            action=action, observation=observation, prev_cache=prev_state.cache
        )

        return cls(
            problem=prev_state.problem,
            prior=prev_state.prior,
            data=cache.data,
            belief=prev_state.belief,
            info=prev_state.info,
            cache=cache,
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
            cache=prev_state.cache,
        )
