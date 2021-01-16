"""Base class for stopping criteria of probabilistic linear solvers."""

from typing import Callable, Optional

import probnum  # pylint: disable="unused-import
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["StoppingCriterion"]

# pylint: disable="invalid-name,too-few-public-methods"


class StoppingCriterion:
    """Stopping criterion of a (probabilistic) linear solver.

    If a stopping criterion returns ``True``, the solver terminates and returns the
    current estimate for the quantities of interest.

    Parameters
    ----------
    stopping_criterion
        Callable defining the stopping criterion.

    See Also
    --------
    MaxIterStoppingCriterion : Stop after a maximum number of iterations.
    ResidualStoppingCriterion : Stop based on the norm of the residual :math:`Ax_i -b`.
    PosteriorStoppingCriterion : Stop based on the uncertainty about the solution.
    """

    def __init__(
        self,
        stopping_criterion: Callable[
            [
                LinearSystem,
                "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
            bool,
        ],
    ):
        self._stopping_criterion = stopping_criterion

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> bool:
        """Evaluate whether the solver has converged.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief : Belief `(x, A, Ainv, b)` over the solution :math:`x`,
                 the system matrix :math:`A`, its inverse :math:`H=A^{-1}` and the
                 right hand side :math:`b`.
        solver_state :
            Current state of the linear solver.
        """
        return self._stopping_criterion(problem, belief, solver_state)
