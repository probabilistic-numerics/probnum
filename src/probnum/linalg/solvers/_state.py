"""State of a probabilistic linear solver."""

import dataclasses
from collections import defaultdict
from typing import Any, List, Mapping, Optional, Tuple

import probnum  # pylint:disable="unused-import"
from probnum import problems

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np


@dataclasses.dataclass
class LinearSolverState:
    r"""State of a probabilistic linear solver.

    The solver state contains the problem to be solved, the current belief over the quantities of interest and any miscellaneous quantities computed during an iteration
    of a probabilistic linear solver. The solver state is passed between the
    different components of the solver and may be used to cache quantities which are used more than once.

    Parameters
    ----------
    problem
        Linear system to be solved.
    prior
        Prior belief over the quantities of interest of the linear system.
    """

    def __init__(
        self,
        problem: problems.LinearSystem,
        prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
    ):
        self.step = 0
        self.has_converged = False

        self.problem = problem
        self.belief = prior

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
