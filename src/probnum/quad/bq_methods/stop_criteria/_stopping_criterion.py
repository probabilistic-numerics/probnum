"""Stopping criteria for Bayesian quadrature."""

from typing import Callable

import numpy as np

from probnum.quad.bq_methods.bq_state import BQState
from probnum.randvars import Normal
from probnum.type import FloatArgType, IntArgType


class StoppingCriterion:
    """Base class for a stopping criterion.

    Parameters
    ----------
    stopping_criterion :
        A function that determines whether a convergence criterion is reached.
    """

    def __init__(self, stopping_criterion: Callable):
        self._stopping_criterion = stopping_criterion

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        """Evaluate the stopping criterion.

        Parameters
        ----------
        integral_belief :
            Current Gaussian belief about the integral.
        bq_state:
            State of the BQ loop.

        Returns
        -------
        has_converged:
            Boolean whether a stopping criterion has been reached
        """
        return self._stopping_criterion(bq_state)


class IntegralVariance(StoppingCriterion):
    """Stop once the integral variance is below some tolerance.

    Parameters
    ----------
    var_tol:
        Tolerance value of the variance.
    """

    def __init__(self, var_tol: FloatArgType = None):
        self.var_tol = var_tol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return integral_belief.var <= self.var_tol


class RelativeError(StoppingCriterion):
    """Stop once the relative change of consecutive integral estimates are smaller than
    a tolerance. That is, the stopping criterion is.

        | (integrals[i] - integrals[i-1]) / integrals[i] | <= tol

    where ``integrals`` holds the BQ integral means.

    Parameters
    ----------
    rel_tol:
        Relative error tolerance on consecutive integral mean values.
    """

    def __init__(self, rel_tol: FloatArgType = None):
        self.rel_tol = rel_tol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return (
            np.abs(
                (integral_belief.mean - bq_state.previous_integral_beliefs[-1].mean)
                / integral_belief.mean
            )
            <= self.rel_tol
        )


class MaxNevals(StoppingCriterion):
    """Stop once a maximum number of iterations is reached.

    Parameters
    ----------
    max_nevals:
        Maximum number of function evaluations.
    """

    def __init__(self, max_nevals: IntArgType = None):
        self.max_nevals = max_nevals
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return bq_state.info.nevals >= self.max_nevals
