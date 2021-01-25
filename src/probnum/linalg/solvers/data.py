"""Data about a linear system collected by a probabilistic linear solver."""

import dataclasses
from collections import namedtuple
from functools import cached_property
from typing import List, Optional, Tuple, Union

import numpy as np

import probnum.linops as linops

# pylint: disable="invalid-name"

__all__ = ["LinearSolverAction", "LinearSolverObservation", "LinearSolverData"]


@dataclasses.dataclass(frozen=True)
class LinearSolverAction:
    r"""Action performed by a linear solver to obtain information about a linear
    system.

    Parameters
    ----------
    A :
        Action to observe the system matrix.
    b :
        Action to observe the right hand side.
    proj :
        Linear projection operator to a (low-dimensional) subspace.
    """
    A: np.ndarray
    b: Optional[np.ndarray] = None
    proj: Optional[Union[linops.LinearOperator, np.ndarray]] = None

    def __eq__(self, other):
        return (
            np.all(self.A == other.A)
            and np.all(self.b == other.b)
            and np.all(self.proj == other.proj)
        )


@dataclasses.dataclass(frozen=True)
class LinearSolverObservation:
    r"""Observation of a linear system.

    Parameters
    ----------
    A :
        Observation of the system matrix.
    b :
        Observation of the right hand side.
    """
    A: np.ndarray
    b: np.ndarray

    def __eq__(self, other):
        return np.all(self.A == other.A) and np.all(self.b == other.b)


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

    def __init__(
        self,
        actions: List[LinearSolverAction],
        observations: List[LinearSolverObservation],
    ):
        if len(actions) != len(observations):
            raise ValueError("Actions and observations must have the same length.")
        self._actions = actions
        self._observations = observations

    def __len__(self):
        """Amount of data collected."""
        return len(self._actions)

    @property
    def actions(self) -> List[LinearSolverAction]:
        """Performed actions by the linear solver."""
        return self._actions

    @property
    def observations(self) -> List[LinearSolverObservation]:
        """Observations of the linear system."""
        return self._observations

    @cached_property
    def actions_arr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Array of performed actions."""
        ActionsArr = namedtuple("ActionsArr", ["A", "b"])

        return ActionsArr(
            np.hstack([action.A for action in self.actions]),
            np.hstack([action.b for action in self.actions]),
        )

    @cached_property
    def observations_arr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tuple of arrays of performed observations of the system matrix and right hand
        side."""
        ObservationsArr = namedtuple("ObservationsArr", ["A", "b"])
        return ObservationsArr(
            np.hstack([observation.A for observation in self.observations]),
            np.hstack([observation.b for observation in self.observations]),
        )
