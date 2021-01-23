"""State of a probabilistic linear solver."""

import dataclasses
from typing import List, Optional

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
    "LinearSolverData",
    "LinearSolverMiscQuantities",
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
    actions: List
    observations: List

    @cached_property
    def actions_arr(self) -> np.ndarray:
        """Array of performed actions."""
        return np.hstack(self.actions)

    @cached_property
    def observations_arr(self) -> np.ndarray:
        """Array of performed observations."""
        return np.hstack(self.observations)


@dataclasses.dataclass
class LinearSolverMiscQuantities:
    r"""Miscellaneous (cached) quantities.

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
    x
        Quantities used to update the belief about the solution.
    A
        Quantities used to update the belief about the system matrix.
    Ainv
        Quantities used to update the belief about the inverse.
    b
        Quantities used to update the belief about the right hand side.
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        x: Optional[
            "probnum.linalg.solvers.belief_updates.LinearSolverBeliefUpdateState"
        ] = None,
        A: Optional[
            "probnum.linalg.solvers.belief_updates.LinearSolverBeliefUpdateState"
        ] = None,
        Ainv: Optional[
            "probnum.linalg.solvers.belief_updates.LinearSolverBeliefUpdateState"
        ] = None,
        b: Optional[
            "probnum.linalg.solvers.belief_updates.LinearSolverBeliefUpdateState"
        ] = None,
    ):
        # pylint: disable="too-many-arguments"

        self.problem = problem
        self.belief = belief
        self.x = x
        self.A = A
        self.Ainv = Ainv
        self.b = b

    @classmethod
    def from_new_data(
        cls,
        action: np.ndarray,
        observation: np.ndarray,
        prev: "LinearSolverMiscQuantities",
    ):
        """Create new miscellaneous cached quantities from new data."""
        new_belief_update_states = {}
        for key, prev_belief_update_state in {
            "x": prev.x,
            "A": prev.A,
            "Ainv": prev.Ainv,
            "b": prev.b,
        }.items():
            if prev_belief_update_state is None:
                new_belief_update_states[key] = None
            else:
                new_belief_update_states[key] = type(
                    prev_belief_update_state
                ).from_new_data(
                    action=action,
                    observation=observation,
                    prev_state=prev_belief_update_state,
                )

        return cls(
            problem=prev.problem,
            belief=prev.belief,
            x=new_belief_update_states["x"],
            A=new_belief_update_states["A"],
            Ainv=new_belief_update_states["Ainv"],
            b=new_belief_update_states["b"],
        )

    @cached_property
    def residual(self) -> np.ndarray:
        r"""Residual :math:`r = A x_i- b` of the solution estimate
        :math:`x_i=\mathbb{E}[\mathsf{x}]` at iteration :math:`i`."""
        try:
            return self.x.residual
        except AttributeError:
            return self.problem.A @ self.belief.x.mean - self.problem.b


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
        Information about the convergence of the linear solver
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
        misc: Optional[LinearSolverMiscQuantities] = None,
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
            else LinearSolverMiscQuantities(problem=problem, belief=belief)
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
        misc = LinearSolverMiscQuantities.from_new_data(
            action=action, observation=observation, prev=prev_state.misc
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
