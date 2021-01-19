"""Abstract base class for belief updates for probabilistic linear solvers."""
import abc
from typing import List, Optional, Tuple

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import dataclasses

import probnum  # pylint: disable="unused-import"
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum._probabilistic_numerical_method import PNMethodData
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["BeliefUpdate", "BeliefUpdateState"]

# pylint: disable="invalid-name,too-many-arguments"


@dataclasses.dataclass
class BeliefUpdateTerms:
    r"""Belief update terms for a quantity of interest.

    Collects the belief update terms for a quantity of interest of a linear
    system, i.e. additive terms for the mean and covariance (factors).
    """
    mean: Optional[linops.LinearOperator] = None
    cov: Optional[linops.LinearOperator] = None
    covfactors: Optional[Tuple[linops.LinearOperator, ...]] = None


@dataclasses.dataclass
class BeliefUpdateState(PNMethodData):
    r"""Quantities computed for the belief update of a linear solver.

    Parameters
    ----------
    action_obs_innerprods
        Inner product(s) :math:`(S^\top Y)_{ij} = s_i^\top y_j` of actions
        and observations. If a vector, actions and observations are assumed to be
        conjugate, i.e. :math:`s_i^\top y_j =0` for :math:`i \neq j`.
    log_rayleigh_quotients
        Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(s_i^\top
        s_i)`.
    step_sizes
        Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer taking
        steps :math:`x_{i+1} = x_i + \alpha_i s_i`.
    """
    x_update: Optional[BeliefUpdateTerms] = None
    A_update: Optional[BeliefUpdateTerms] = None
    Ainv_update: Optional[BeliefUpdateTerms] = None
    b_update: Optional[BeliefUpdateTerms] = None
    action_obs_innerprods: Optional[List[float]] = None
    log_rayleigh_quotients: Optional[List[float]] = None
    step_sizes: Optional[List[float]] = None


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
    action :
        Action(s) to probe the linear system with.
    observation :
        Observation(s) of the linear system for the given action(s).
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
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ):
        self.problem = problem
        self.belief = belief
        if action.ndim == 1:
            self.action = action[:, None]
        else:
            self.action = action
        if observation.ndim == 1:
            self.observation = observation[:, None]
        else:
            self.observation = observation
        self._x = None
        self._Ainv = None
        self._A = None
        self._b = None
        self.solver_state = solver_state

    def precompute(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> Tuple[BeliefUpdateState, Optional["probnum.linalg.solvers.LinearSolverState"]]:
        """Pre-compute quantities for the belief update.

        This function pre-computes necessary quantities for the belief update. This is
        useful to efficiently perform hyperparameter optimization prior to actually
        performing the update.

        Parameters
        ----------
        problem
        belief
        action
        observation
        solver_state

        Returns
        -------
        """
        raise NotImplementedError

    def update(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
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
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError

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
