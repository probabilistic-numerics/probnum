"""State of a probabilistic linear solver."""

from __future__ import annotations

from collections import defaultdict
import dataclasses
from typing import Any, DefaultDict, List, Optional, Tuple

import numpy as np

import probnum  # pylint:disable="unused-import"
from probnum import problems


@dataclasses.dataclass
class LinearSolverState:
    """State of a probabilistic linear solver.

    The solver state separates the state of a probabilistic linear solver from the
    algorithm itself, making the solver stateless.
    The state contains the problem to be solved, the current belief over the quantities
    of interest and any miscellaneous quantities computed during an iteration of
    a probabilistic linear solver.
    The solver state is passed between the different components of the solver and
    may be used internally to cache quantities which are used more than once.

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
        self._problem: problems.LinearSystem = problem

        # Belief
        self._prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief" = prior
        self._belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief" = prior

        # Caches
        self._actions: List[np.ndarray] = [None]
        self._observations: List[Any] = [None]
        self._residuals: List[np.ndarray] = [
            self.problem.b - self.problem.A @ self.belief.x.mean,
        ]
        self.cache: DefaultDict[str, Any] = defaultdict(list)

        # Solver info
        self._step: int = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(step={self.step})"

    @property
    def problem(self) -> problems.LinearSystem:
        """Linear system to be solved."""
        return self._problem

    @property
    def prior(self) -> "probnum.linalg.solvers.beliefs.LinearSystemBelief":
        """Prior belief over the quantities of interest of the linear system."""
        return self._prior

    @property
    def step(self) -> int:
        """Current step of the solver."""
        return self._step

    @property
    def belief(self) -> "probnum.linalg.solvers.beliefs.LinearSystemBelief":
        """Belief over the quantities of interest of the linear system."""
        return self._belief

    @belief.setter
    def belief(
        self, belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief"
    ) -> None:
        self._belief = belief

    @property
    def action(self) -> Optional[np.ndarray]:
        """Action of the solver for the current step.

        Is ``None`` at the beginning of a step and will be set by the policy.
        """
        return self._actions[self.step]

    @action.setter
    def action(self, value: np.ndarray) -> None:
        assert self._actions[self.step] is None
        self._actions[self.step] = value

    @property
    def observation(self) -> Optional[Any]:
        """Observation of the solver for the current step.

        Is ``None`` at the beginning of a step, will be set by the observation model for
        a given action.
        """
        return self._observations[self.step]

    @observation.setter
    def observation(self, value: Any) -> None:
        assert self._observations[self.step] is None
        self._observations[self.step] = value

    @property
    def actions(self) -> Tuple[np.ndarray, ...]:
        """Actions taken by the solver."""
        return tuple(self._actions)

    @property
    def observations(self) -> Tuple[Any, ...]:
        """Observations of the problem by the solver."""
        return tuple(self._observations)

    @property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r_{i} = b - Ax_{i}`."""
        if self._residuals[self.step] is None:
            self._residuals[self.step] = (
                self.problem.b - self.problem.A @ self.belief.x.mean
            )
        return self._residuals[self.step]

    @property
    def residuals(self) -> Tuple[np.ndarray, ...]:
        r"""Residuals :math:`\{b - Ax_i\}_i`."""
        return tuple(self._residuals)

    def next_step(self) -> None:
        """Advance the solver state to the next solver step.

        Called after a completed step / iteration of the linear solver.
        """
        self._actions.append(None)
        self._observations.append(None)
        self._residuals.append(None)

        self._step += 1
