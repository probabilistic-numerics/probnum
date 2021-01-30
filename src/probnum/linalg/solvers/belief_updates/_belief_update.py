"""Abstract base class for belief updates for probabilistic linear solvers."""
import abc
from typing import Optional, Tuple, Type

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum  # pylint: disable="unused-import"
import probnum.random_variables as rvs
from probnum.linalg.solvers._state import LinearSolverCache, LinearSolverState
from probnum.linalg.solvers.beliefs import (
    LinearSystemBelief,
    NoisySymmetricNormalLinearSystemBelief,
    SymmetricNormalLinearSystemBelief,
)
from probnum.linalg.solvers.data import (
    LinearSolverAction,
    LinearSolverData,
    LinearSolverObservation,
)
from probnum.linalg.solvers.observation_ops import ObservationOp
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSolverBeliefUpdate", "LinearSolverQoIBeliefUpdate"]

# pylint: disable="invalid-name,too-many-arguments"


class LinearSolverQoIBeliefUpdate(abc.ABC):
    """Belief update for a quantity of interest."""

    def __init__(
        self,
        prior: LinearSystemBelief,
    ):
        self.prior = prior

    def __call__(
        self,
        problem: LinearSystem,
        hyperparams: "probnum.linalg.solvers.hyperparams.LinearSolverHyperparams",
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> rvs.RandomVariable:
        """Update the belief about the quantity of interest given observations.

        Parameters
        ----------
        problem :
            Linear system to solve.
        hyperparams :
            Hyperparameters of the belief.
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError


class LinearSolverBeliefUpdate(abc.ABC):
    r"""Belief update of a probabilistic linear solver.

    Computes the updated beliefs over quantities of interest of a linear system after
    making observations about the system given a prior belief.

    Parameters
    ---------
    prior :
    x_belief_update_type :
    A_belief_update_type :
    Ainv_belief_update_type :
    b_belief_update_type :

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given a symmetric
        matrix-variate normal belief and linear observations.
    """

    def __init__(
        self,
        prior: LinearSystemBelief,
        cache_type: Type[LinearSolverCache],
        x_belief_update_type: Type[LinearSolverQoIBeliefUpdate],
        A_belief_update_type: Type[LinearSolverQoIBeliefUpdate],
        Ainv_belief_update_type: Type[LinearSolverQoIBeliefUpdate],
        b_belief_update_type: Type[LinearSolverQoIBeliefUpdate],
    ):
        self._prior = prior
        self.cache_type = cache_type
        self._x_belief_update = x_belief_update_type(prior=prior)
        self._A_belief_update = A_belief_update_type(prior=prior)
        self._Ainv_belief_update = Ainv_belief_update_type(prior=prior)
        self._b_belief_update = b_belief_update_type(prior=prior)

    def __call__(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        action: LinearSolverAction,
        observation: LinearSolverObservation,
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
        belief :
            Belief over the quantities of interest
        hyperparams :
            Hyperparameters of the belief.
        action :
            Action to probe the linear system with.
        observation :
            Observation of the linear system for the given action.
        solver_state :
            Current state of the linear solver.
        """
        if solver_state is None:

            solver_state = LinearSolverState(
                problem=problem,
                prior=self._prior,
                belief=belief,
                data=LinearSolverData(
                    actions=[action],
                    observations=[observation],
                ),
                cache=self.cache_type.from_new_data(
                    action=action,
                    observation=observation,
                    prev_cache=self.cache_type(
                        problem=problem,
                        prior=self._prior,
                        belief=self._prior,
                    ),
                ),
            )

        # Update belief (using optimized hyperparameters)
        updated_belief = type(self._prior)(
            x=self._x_belief_update(
                problem=problem,
                hyperparams=hyperparams,
                solver_state=solver_state,
            ),
            Ainv=self._Ainv_belief_update(
                problem=problem,
                hyperparams=hyperparams,
                solver_state=solver_state,
            ),
            A=self._A_belief_update(
                problem=problem,
                hyperparams=hyperparams,
                solver_state=solver_state,
            ),
            b=self._b_belief_update(
                problem=problem,
                hyperparams=hyperparams,
                solver_state=solver_state,
            ),
            hyperparams=hyperparams,
        )

        # Create new solver state from updated belief
        updated_solver_state = LinearSolverState.from_updated_belief(
            updated_belief=updated_belief, prev_state=solver_state
        )

        return updated_belief, updated_solver_state
