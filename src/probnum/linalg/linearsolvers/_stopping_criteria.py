"""Stopping criteria of probabilistic linear solvers."""

import warnings
from typing import Callable

import numpy as np

import probnum
from probnum.problems import LinearSystem
from probnum.type import IntArgType, ScalarArgType

# pylint: disable="invalid-name,too-few-public-methods"


class StoppingCriterion:
    """Stopping criterion of a (probabilistic) linear solver.

    If a stopping criterion returns ``True``, the solver terminates and returns the
    current estimate for the quantities of interest.

    Parameters
    ----------
    stopping_criterion
        Callable defining the stopping criterion.
    """

    def __init__(
        self,
        stopping_criterion: Callable[
            [
                LinearSystem,
                "probnum.linalg.linearsolvers.LinearSolverState",
            ],
            bool,
        ],
    ):
        self._stopping_criterion = stopping_criterion

    def __call__(
        self,
        problem: LinearSystem,
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> bool:
        """Evaluate whether the solver has converged.

        Parameters
        ----------
        problem :
            Linear system to solve.
        solver_state :
            Current state of the linear solver.
        """
        return self._stopping_criterion(problem, solver_state)


class MaxIterStoppingCriterion(StoppingCriterion):
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
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> bool:
        if self.maxiter is None:
            _maxiter = problem.A.shape[0] * 10
        else:
            _maxiter = self.maxiter

        if solver_state.iteration >= _maxiter:
            warnings.warn(
                "Iteration terminated. Solver reached the maximum number of iterations."
            )
            return True
        else:
            return False


class ResidualStoppingCriterion(StoppingCriterion):
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
    """

    def __init__(self, atol: ScalarArgType = 10 ** -5, rtol: ScalarArgType = 10 ** -5):
        self.atol = atol
        self.rtol = rtol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> bool:
        # Compute residual
        x, _, _, _ = solver_state.belief
        resid = problem.A @ x.mean.reshape(-1, 1) - problem.b.reshape(-1, 1)
        resid_norm = np.linalg.norm(resid)

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b)
        return (resid_norm <= self.atol) or (resid_norm <= self.rtol * b_norm)


class PosteriorStoppingCriterion(StoppingCriterion):
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
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> bool:
        # Trace of the solution covariance
        x, _, _, _ = solver_state.belief
        # TODO: replace this with (existing) more efficient trace computation; maybe an
        #  iterative update to the trace property of the linear operator
        trace_sol_cov = x.cov.trace()

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b)
        return (np.abs(trace_sol_cov) <= self.atol ** 2) or (
            np.abs(trace_sol_cov) <= (self.rtol * b_norm) ** 2
        )
