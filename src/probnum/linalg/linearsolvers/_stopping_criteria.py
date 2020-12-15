"""Stopping criteria of probabilistic linear solvers."""

import warnings
from typing import Callable, Tuple, Union

import numpy as np

import probnum.random_variables as rvs
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
                int,
                LinearSystem,
                Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
            ],
            Tuple[bool, Union[str, None]],
        ],
    ):
        self._stopping_criterion = stopping_criterion

    def __call__(
        self,
        iteration: int,
        problem: LinearSystem,
        belief: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
    ) -> Tuple[bool, Union[str, None]]:
        """Evaluate whether the solver has converged.

        Parameters
        ----------
        iteration
            Current iteration of the solver.
        problem :
            Linear system to solve.
        belief :
            Belief over the parameters :code:`(x, A, Ainv)` of the linear system.
        """
        return self._stopping_criterion(iteration, problem, belief)


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
        iteration: int,
        problem: LinearSystem,
        belief: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
    ) -> Tuple[bool, Union[str, None]]:
        if self.maxiter is None:
            _maxiter = problem.A.shape[0] * 10
        else:
            _maxiter = self.maxiter

        if iteration >= _maxiter:
            warnings.warn(
                "Iteration terminated. Solver reached the maximum number of iterations."
            )
            return True, self.__class__.__name__
        else:
            return False, None


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
    """

    def __init__(self, atol: ScalarArgType = 10 ** -5, rtol: ScalarArgType = 10 ** -5):
        self.atol = atol
        self.rtol = rtol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(
        self,
        iteration: int,
        problem: LinearSystem,
        belief: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
    ) -> Tuple[bool, Union[str, None]]:
        # Compute residual
        x, _, _ = belief
        resid = problem.A @ x.mean.reshape(-1, 1) - problem.b.reshape(-1, 1)
        resid_norm = np.linalg.norm(resid)

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b)
        if resid_norm <= self.atol:
            return True, self.__class__.__name__ + "_atol"
        elif resid_norm <= self.rtol * b_norm:
            return True, self.__class__.__name__ + "_rtol"
        else:
            return False, None


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
        iteration: int,
        problem: LinearSystem,
        belief: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
    ) -> Tuple[bool, Union[str, None]]:
        # Trace of the solution covariance
        x, _, _ = belief
        # TODO: replace this with (existing) more efficient trace computation; maybe an
        #  iterative update to the trace property of the linear operator
        trace_sol_cov = x.cov.trace()

        # Compare (relative) residual to tolerances
        b_norm = np.linalg.norm(problem.b)
        if np.abs(trace_sol_cov) <= self.atol ** 2:
            return True, self.__class__.__name__ + "_atol"
        elif np.abs(trace_sol_cov) <= (self.rtol * b_norm) ** 2:
            return True, self.__class__.__name__ + "_rtol"
        else:
            return False, None
