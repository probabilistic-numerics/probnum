"""Stopping criteria of probabilistic linear solvers."""

from typing import Callable, Optional

import numpy as np

import probnum  # pylint: disable="unused-import
from probnum.problems import LinearSystem
from probnum.type import IntArgType, ScalarArgType

# Public classes and functions. Order is reflected in documentation.
__all__ = ["StoppingCriterion", "MaxIterations", "Residual", "PosteriorContraction"]

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


class MaxIterations(StoppingCriterion):
    """Maximum number of iterations.

    Stop when a maximum number of iterations is reached. If none is
    specified, defaults to :math:`10n`, where :math:`n` is the dimension
    of the solution to the linear system.
    """

    def __init__(self, maxiter: IntArgType = None):
        self.maxiter = maxiter
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> bool:
        if self.maxiter is None:
            _maxiter = problem.A.shape[0] * 10
        else:
            _maxiter = self.maxiter

        try:
            return solver_state.iteration >= _maxiter
        except AttributeError:
            return False


class Residual(StoppingCriterion):
    """Residual stopping criterion.

    Terminate when the norm of the residual :math:`r_{i} = A x_{i} - b` is
    sufficiently small, i.e. if it satisfies :math:`\\lVert r_i \\rVert \\leq \\max(
    \\text{atol}, \\text{rtol} \\lVert b \\rVert)`.

    Parameters
    ----------
    atol :
        Absolute residual tolerance.
    rtol :
        Relative residual tolerance.
    norm_ord :
        Order of the norm. Defaults to the euclidean (:math:`p=2`) norm. See
        :func:`numpy.linalg.norm` for a complete list of available choices.
    """

    def __init__(
        self, atol: ScalarArgType = 10 ** -5, rtol: ScalarArgType = 10 ** -5, norm_ord=2
    ):
        self.atol = atol
        self.rtol = rtol
        self.norm_ord = norm_ord
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> bool:
        # Compute residual norm
        try:
            residual = solver_state.residual
        except AttributeError:
            residual = problem.A @ belief.x.mean - problem.b
        residual_norm = np.linalg.norm(residual.flatten(), ord=self.norm_ord)

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b.flatten(), ord=self.norm_ord)
        return residual_norm <= self.atol or residual_norm <= self.rtol * b_norm


class PosteriorContraction(StoppingCriterion):
    """Posterior contraction stopping criterion.

    Terminate when the uncertainty about the solution is sufficiently small, i.e. if it
    satisfies :math:`\\sqrt{\\operatorname{tr}(\\mathbb{Cov}(\\mathsf{x}))}
    \\leq \\max(\\text{atol}, \\text{rtol} \\lVert b \\rVert)`.

    Parameters
    ----------
    atol :
        Absolute residual tolerance.
    rtol :
        Relative residual tolerance.
    """

    def __init__(self, atol: ScalarArgType = 10 ** -5, rtol: ScalarArgType = 10 ** -5):
        self.atol = atol
        self.rtol = rtol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> bool:
        # Trace of the solution covariance
        trace_sol_cov = belief.x.cov.trace()

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b.flatten())
        return (
            np.abs(trace_sol_cov) <= self.atol ** 2
            or np.abs(trace_sol_cov) <= (self.rtol * b_norm) ** 2
        )
