"""Abstract base class for belief updates for probabilistic linear solvers."""
import abc
from typing import Optional, Tuple

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum  # pylint: disable="unused-import"
import probnum.random_variables as rvs
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["BeliefUpdate"]

# pylint: disable="invalid-name,too-many-arguments"


class BeliefUpdate(abc.ABC):
    r"""Belief update of a probabilistic linear solver.

    Computes the updated beliefs over quantities of interest of a linear system after
    making observations about the system given a prior belief.

    Parameters
    ----------
    problem :
        Linear system to solve.
    belief :
        Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
        linear system.
    actions :
        Actions to probe the linear system with.
    observations :
        Observations of the linear system for the given actions.
    solver_state :
        Current state of the linear solver.

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given a symmetric
        matrix-variate normal belief and linear observations.
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        actions: np.ndarray,
        observations: np.ndarray,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ):
        self.problem = problem
        self.belief = belief
        if actions.ndim == 1:
            self.actions = actions[:, None]
        else:
            self.actions = actions
        if observations.ndim == 1:
            self.observations = observations[:, None]
        else:
            self.observations = observations
        self._x = None
        self._Ainv = None
        self._A = None
        self._b = None
        self.solver_state = solver_state

    def __call__(
        self,
    ) -> Tuple[
        rvs.RandomVariable,
        rvs.RandomVariable,
        rvs.RandomVariable,
        rvs.RandomVariable,
        Optional["probnum.linalg.solvers.LinearSolverState"],
    ]:
        """Update the belief about the quantities of interest of the linear system."""
        return (
            self.x,
            self.Ainv,
            self.A,
            self.b,
            self.solver_state,
        )

    @cached_property
    def x(self) -> rvs.RandomVariable:
        """Updated belief about the solution :math:`x` of the linear system."""
        raise NotImplementedError

    @cached_property
    def A(self) -> rvs.RandomVariable:
        """Updated belief about the system matrix :math:`A`."""
        raise NotImplementedError

    @cached_property
    def Ainv(self) -> rvs.RandomVariable:
        """Updated belief about the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    @cached_property
    def b(self) -> rvs.RandomVariable:
        """Updated belief about the right hand side :math:`b` of the linear system."""
        raise NotImplementedError
