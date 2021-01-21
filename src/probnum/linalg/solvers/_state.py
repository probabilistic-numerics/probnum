import dataclasses
from typing import Callable, List, Optional

import numpy as np

from probnum.linalg.solvers import belief_updates, beliefs, stop_criteria

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from probnum.problems import LinearSystem

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
    stopping_criterion: Optional[List[stop_criteria.StoppingCriterion]] = None


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
    x
        Quantities used to update the belief about the solution.
    A
        Quantities used to update the belief about the system matrix.
    Ainv
        Quantities used to update the belief about the inverse.
    b
        Quantities used to update the belief about the right hand side.
    """
    x: belief_updates.LinearSolverBeliefUpdate
    A: belief_updates.LinearSolverBeliefUpdate
    Ainv: belief_updates.LinearSolverBeliefUpdate
    b: belief_updates.LinearSolverBeliefUpdate


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
        prior: beliefs.LinearSystemBelief,
        data: LinearSolverData,
        belief: beliefs.LinearSystemBelief,
        info: Optional[LinearSolverInfo] = None,
        misc: Optional[LinearSolverMiscQuantities] = None,
    ):

        self.problem = problem
        self.prior = prior
        self.data = data
        self.belief = belief
        if info is None:
            self.info = LinearSolverInfo()
        else:
            self.info = info
        if misc is None:
            self.misc = LinearSolverMiscQuantities(
                iteration=self.info.iteration,
                problem=problem,
                belief=belief,
                data=data,
            )
        else:
            self.misc = misc
