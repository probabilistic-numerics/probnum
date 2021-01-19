"""Abstract base class for belief updates for probabilistic linear solvers."""
import abc
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import dataclasses

import probnum  # pylint: disable="unused-import"
import probnum.linops as linops
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
class BeliefUpdateState:
    r"""Quantities computed for the belief update of a linear solver.

    Parameters
    ----------
    x
        Belief update term for the solution.
    A
        Belief update term for the system matrix.
    Ainv
        Belief update term for the inverse.
    b
        Belief update term for the right hand side.
    step_sizes
        Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer taking
        steps :math:`x_{i+1} = x_i + \alpha_i s_i`.
    """
    x: Optional[BeliefUpdateTerms] = None
    A: Optional[BeliefUpdateTerms] = None
    Ainv: Optional[BeliefUpdateTerms] = None
    b: Optional[BeliefUpdateTerms] = None
    step_sizes: Optional[List[float]] = None


class BeliefUpdate(abc.ABC):
    r"""Belief update of a probabilistic linear solver.

    Computes the updated beliefs over quantities of interest of a linear system after
    making observations about the system given a prior belief.

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given a symmetric
        matrix-variate normal belief and linear observations.
    """

    def update(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        action: np.ndarray,
        observation: np.ndarray,
        hyperparams: Optional["probnum.PNMethodHyperparams"] = None,
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
        raise NotImplementedError
