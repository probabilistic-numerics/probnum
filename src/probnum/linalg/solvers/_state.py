"""State of a probabilistic linear solver."""

import dataclasses
from typing import Any, List, Optional, Tuple

import numpy as np

import probnum  # pylint:disable="unused-import"
from probnum import problems


@dataclasses.dataclass
class ProbabilisticLinearSolverState:
    """State of a probabilistic linear solver.

    The solver state separates the state of a probabilistic linear solver from the algorithm itself, making the solver stateless. The state contains the problem to be solved, the current belief over the quantities of interest and any miscellaneous quantities computed during an iteration of a probabilistic linear solver. The solver state is passed between the different components of the solver and may be used internally to cache quantities which are used more than once.

    Parameters
    ----------
    problem
        Linear system to be solved.
    prior
        Prior belief over the quantities of interest of the linear system.
    rng
        Random number generator.
    """

    def __init__(
        self,
        problem: problems.LinearSystem,
        prior: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        rng: Optional[np.random.Generator] = None,
    ):
        self.problem = problem
        self.belief = prior

        self.step = 0

        self._actions: List[np.ndarray] = [None]
        self._observations: List[Any] = [None]
        self._residuals: List[np.ndarray] = [
            self.problem.A @ self.belief.x.mean - self.problem.b,
            None,
        ]
        self.rng = rng

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(step={self.step})"

    @property
    def action(self) -> Optional[np.ndarray]:
        """Action of the solver for the current step.

        Is ``None`` at the beginning of a step and will be set by the
        policy.
        """
        return self._actions[self.step]

    @action.setter
    def action(self, value: np.ndarray) -> None:
        assert self._actions[self.step] is None
        self._actions[self.step] = value

    @property
    def observation(self) -> Optional[Any]:
        """Observation of the solver for the current step.

        Is ``None`` at the beginning of a step, will be set by the
        observation model for a given action.
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
        r"""Cached residual :math:`Ax_i-b` for the current solution estimate :math:`x_i`."""
        if self._residuals[self.step] is None:
            self._residuals[self.step] = (
                self.problem.A @ self.belief.x.mean - self.problem.b
            )
        return self._residuals[self.step]

    @property
    def residuals(self) -> Tuple[np.ndarray, ...]:
        r"""Residuals :math:`\{Ax_i - b\}_i`."""
        return tuple(self._residuals)

    def next_step(self) -> None:
        """Advance the solver state to the next solver step.

        Called after a completed step / iteration of the probabilistic
        linear solver.
        """
        self._actions.append(None)
        self._observations.append(None)
        self._residuals.append(None)

        self.step += 1
