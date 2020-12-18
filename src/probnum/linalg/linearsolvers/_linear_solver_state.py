"""State of a probabilistic linear solver on a given problem."""

import dataclasses
from typing import List, Optional, Union

import numpy as np

import probnum.random_variables as rvs
from probnum._probabilistic_numerical_method import PNMethodState


@dataclasses.dataclass
class LinearSolverState(PNMethodState):
    r"""State of a probabilistic linear solver.

    The solver state contains miscellaneous quantities computed during an iteration
    of a probabilistic linear solver. The solver state is passed between the
    different components of the solver and may be used by them.

    For example the residual :math:`r_i = Ax_i - b` can (depending on the prior) be
    updated more efficiently than in :math:`\mathcal{O}(n^2)` and is therefore part
    of the solver state and passed to the stopping criteria.

    Parameters
    ----------
    belief
        Current belief over the solution :math:`x`, the system matrix :math:`A`, its
        inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
    actions
        Performed actions :math:`s_i`.
    observations
        Collected observations :math:`y_i = A s_i`.
    iteration
        Current iteration :math:`i` of the solver.
    residual
        Residual :math:`r_i = Ax_i - b` of the current solution.
    rayleigh_quotients
        Rayleigh quotients :math:`R(A, s_i) = \frac{s_i^\top A s_i}{s_i^\top s_i}`.
    has_converged
        Has the solver converged?
    stopping_criterion
        Stopping criterion which caused termination of the solver.

    Examples
    --------

    """

    actions: List[np.ndarray]
    observations: List[np.ndarray]
    iteration: int = 0
    residual: Optional[Union[np.ndarray, rvs.RandomVariable]] = None
    rayleigh_quotients: Optional[List[float]] = None
    has_converged: bool = False
    stopping_criterion: Optional[List["StoppingCriterion"]] = None
