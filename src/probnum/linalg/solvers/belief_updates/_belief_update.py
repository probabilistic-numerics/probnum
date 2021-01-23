"""Abstract base class for belief updates for probabilistic linear solvers."""
import abc
from typing import Optional, Tuple, Type

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum  # pylint: disable="unused-import"
import probnum.random_variables as rvs
from probnum.linalg.solvers._state import (
    LinearSolverData,
    LinearSolverMiscQuantities,
    LinearSolverState,
)
from probnum.linalg.solvers.beliefs import (
    LinearSystemBelief,
    NoisySymmetricNormalLinearSystemBelief,
    SymmetricNormalLinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.linalg.solvers.observation_ops import MatVecObservation
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverBeliefUpdate",
    "LinearSolverBeliefUpdateState",
]

# pylint: disable="invalid-name,too-many-arguments"


class LinearSolverBeliefUpdateState(abc.ABC):
    r"""Belief update state.

    State containing quantities which are used during the belief update of a
    probabilistic linear solver.

    Parameters
    ----------
    problem :
        Linear system to solve.
    prior :
        Prior belief about the quantities of interest.
    belief :
        Current belief about the quantities of interest.
    action :
        Action taken by the solver given by its policy.
    observation :
        Observation of the linear system for the corresponding action.
    prev_state :
        Previous belief update state prior to the new observation.
    """

    def __init__(
        self,
        problem: LinearSystem,
        prior: LinearSystemBelief,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams"
        ] = None,
        prev_state: Optional["LinearSolverBeliefUpdateState"] = None,
    ):

        self.problem = problem
        self.prior = prior
        self.belief = belief
        self.action = action
        self.observation = observation
        self.hyperparams = hyperparams
        self.prev_state = prev_state

    @classmethod
    def from_new_data(
        cls,
        action: np.ndarray,
        observation: np.ndarray,
        prev_state: "LinearSolverBeliefUpdateState",
    ):
        """Create a new belief update state from a previous one and newly observed
        data."""
        return cls(
            problem=prev_state.problem,
            prior=prev_state.prior,
            belief=prev_state.belief,
            action=action,
            observation=observation,
            hyperparams=prev_state.hyperparams,
            prev_state=prev_state,
        )

    def updated_belief(
        self,
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams"
        ] = None,
    ) -> rvs.RandomVariable:
        """Updated belief about the quantity of interest."""
        raise NotImplementedError


class LinearSolverBeliefUpdate(abc.ABC):
    r"""Belief update of a probabilistic linear solver.

    Computes the updated beliefs over quantities of interest of a linear system after
    making observations about the system given a prior belief.

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given a symmetric
        matrix-variate normal belief and linear observations.
    """

    def __init__(
        self,
        x_belief_update_state_type: Type[LinearSolverBeliefUpdateState],
        A_belief_update_state_type: Type[LinearSolverBeliefUpdateState],
        Ainv_belief_update_state_type: Type[LinearSolverBeliefUpdateState],
        b_belief_update_state_type: Type[LinearSolverBeliefUpdateState],
    ):
        self._x_update_state_type = x_belief_update_state_type
        self._A_update_state_type = A_belief_update_state_type
        self._Ainv_update_state_type = Ainv_belief_update_state_type
        self._b_update_state_type = b_belief_update_state_type

    def update_belief(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional[
            "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams"
        ] = None,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> Tuple[
        LinearSystemBelief, Optional["probnum.linalg.solvers.LinearSolverState"]
    ]:
        """Update the belief given observations.

        Parameters
        ----------
        problem :
            Linear system to solve.
        action :
            Action to probe the linear system with.
        observation :
            Observation of the linear system for the given action.
        hyperparams :
            Hyperparameters of the belief.
        solver_state :
            Current state of the linear solver.
        """
        if solver_state is None:

            update_states = {}
            for key, update_state_types in {
                "x": self._x_update_state_type,
                "A": self._A_update_state_type,
                "Ainv": self._Ainv_update_state_type,
                "b": self._b_update_state_type,
            }.items():
                update_states[key] = update_state_types(
                    problem=problem,
                    prior=belief,
                    belief=belief,
                    action=action,
                    observation=observation,
                    hyperparams=hyperparams,
                )

            solver_state = LinearSolverState(
                problem=problem,
                prior=belief,
                belief=belief,
                data=LinearSolverData(
                    actions=[action],
                    observations=[observation],
                ),
                misc=LinearSolverMiscQuantities(
                    problem=problem,
                    belief=belief,
                    x=update_states["x"],
                    A=update_states["A"],
                    Ainv=update_states["Ainv"],
                    b=update_states["b"],
                ),
            )

        # Update belief (using optimized hyperparameters)
        updated_belief = LinearSystemBelief(
            x=solver_state.misc.x.updated_belief(hyperparams=hyperparams),
            Ainv=solver_state.misc.Ainv.updated_belief(hyperparams=hyperparams),
            A=solver_state.misc.A.updated_belief(hyperparams=hyperparams),
            b=solver_state.misc.b.updated_belief(hyperparams=hyperparams),
        )

        # Create new solver state from updated belief
        updated_solver_state = LinearSolverState.from_updated_belief(
            updated_belief=updated_belief, prev_state=solver_state
        )

        return updated_belief, updated_solver_state
