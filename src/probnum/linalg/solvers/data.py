"""Data about a linear system collected by a probabilistic linear solver."""

import dataclasses
from collections import namedtuple
from functools import cached_property
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy

import probnum.linops as linops

# pylint: disable="invalid-name"

__all__ = ["LinearSolverAction", "LinearSolverObservation", "LinearSolverData"]


@dataclasses.dataclass(frozen=True)
class LinearSolverAction:
    r"""Action performed by a linear solver to obtain information about a linear
    system.

    Parameters
    ----------
    actA :
        Action to observe the system matrix.
    actb :
        Action to observe the right hand side.
    proj :
        Linear projection operator to a (low-dimensional) subspace.
    """
    actA: np.ndarray
    actb: Optional[np.ndarray] = None
    proj: Optional[Union[linops.LinearOperator, np.ndarray]] = None

    def __eq__(self, other):
        return (
            np.all(self.actA == other.actA)
            and np.all(self.actb == other.actb)
            and np.all(self.proj == other.proj)
        )


@dataclasses.dataclass(frozen=True)
class LinearSolverObservation:
    r"""Observation of a linear system.

    Parameters
    ----------
    obsA :
        Observation of the system matrix for a given action.
    obsb :
        Observation of the right hand side for a given action.
    A :
        Observed system matrix or linear operator.
    b :
        Observed right hand side.
    """
    obsA: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator]
    obsb: np.ndarray
    A: Optional[Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator]] = None
    b: Optional[np.ndarray] = None

    def __eq__(self, other):
        return (
            np.all(self.obsA == other.obsA)
            and np.all(self.obsb == other.obsb)
            and np.all(self.A == other.A)
            and np.all(self.b == other.b)
        )


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
        ActionsArr = namedtuple("ActionsArr", ["actA", "actb"])

        arr_actA = np.hstack([action.actA for action in self.actions])
        arr_actb = np.hstack([action.actb for action in self.actions])

        if arr_actA[0] is None:
            arr_actA = np.array([[None]])
        if arr_actb[0] is None:
            arr_actb = np.array([[None]])
        return ActionsArr(actA=arr_actA, actb=arr_actb)

    @cached_property
    def observations_arr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tuple of arrays of performed observations of the system matrix and right hand
        side."""
        ObservationsArr = namedtuple("ObservationsArr", ["obsA", "obsb"])
        return ObservationsArr(
            obsA=np.hstack([observation.obsA for observation in self.observations]),
            obsb=np.hstack([observation.obsb for observation in self.observations]),
        )

    @classmethod
    def from_arrays(
        cls,
        actions_arr: Tuple[np.ndarray, np.ndarray],
        observations_arr: Tuple[np.ndarray, np.ndarray],
    ):
        """Create a linear solver data object from actions and observations given as
        tuples of arrays."""
        return cls(
            actions=[
                LinearSolverAction(actA=actions_arr[0][:, i], actb=actions_arr[1][:, i])
                for i in range(actions_arr[0].shape[1])
            ],
            observations=[
                LinearSolverObservation(
                    obsA=observations_arr[0][:, i], obsb=observations_arr[1][:, i]
                )
                for i in range(observations_arr[0].shape[1])
            ],
        )
