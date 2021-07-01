"""State of a probabilistic linear solver."""

import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, List, Mapping, Optional, Tuple

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum.linalg.solvers.data import LinearSolverData
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

    problem
        Linear system to be solved.
    prior
        Prior belief over the quantities of interest of the linear system.
    """

    def __init__(
        self,
        problem: LinearSystem,
        prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
    ):
        # pylint: disable="too-many-arguments"

        self.step = 0
        self.has_converged = False

        self.problem = problem
        self.belief = self.prior

        self._actions: List[np.ndarray] = [None]
        self._observations: List[Any] = [None]
        self._residuals: List[np.ndarray] = [
            self.problem.A @ self.belief.x.mean - self.problem.b
        ]

        self._cache = defaultdict(list)

    @property
    def action(self) -> Optional[np.ndarray]:
        """Most recent action of the solver."""
        return self.actions[self.step]

    @action.setter  # TODO: this should really be private
    def action(self, value: np.ndarray) -> None:
        assert self._actions[self.step] is None
        self._actions[self.step] = value

    @property
    def observation(self) -> Optional[Any]:
        """Most recent observation of the solver.

        Is `None` at the beginning of a step, will be set by the
        observation model for a given action.
        """
        return self.observations[self.step]

    @observation.setter  # TODO: this should really be private
    def observation(self, value: Any) -> None:
        assert self._observations[self.step] is None
        self._observations[self.step] = value

    @property
    def actions(self) -> Tuple[np.ndarray, ...]:
        """Actions taken by the solver."""
        return tuple(self._actions[:-1])

    @property
    def observations(self) -> Tuple[Any, ...]:
        """Observations of the problem by the solver."""
        return tuple(self._observations[:-1])

    @property
    def residual(self) -> np.ndarray:
        return self.residuals[self.step]

    @property
    def residuals(self) -> Tuple[np.ndarray, ...]:
        return tuple(self._residuals[:-1])

    @property
    def cache(self) -> Mapping[str, Any]:
        """Dynamic cache."""
        return self._cache

    def next_step(self):
        """"""
        self._actions.append(None)
        self._observations.append(None)

        self.step += 1
