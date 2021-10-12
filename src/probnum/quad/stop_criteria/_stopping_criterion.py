"""Stopping criteria for Bayesian quadrature."""


import numpy as np

from probnum.quad.bq_methods.bq_state import BQState
from probnum.randvars import Normal
from probnum.type import FloatArgType, IntArgType


class StoppingCriterion:
    """Base class for a stopping criterion."""

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
        raise NotImplementedError


class IntegralVarianceTolerance(StoppingCriterion):
    """Stop once the integral variance is below some tolerance.

    Parameters
    ----------
    var_tol:
        Tolerance value of the variance.
    """

    def __init__(self, var_tol: FloatArgType):
        self.var_tol = var_tol

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return integral_belief.var <= self.var_tol


class RelativeMeanChange(StoppingCriterion):
    """Stop once the relative change of consecutive integral estimates are smaller than
    a tolerance. That is, the stopping criterion is.

        | current_integral_estimate - previous_integral_estimate) /
      current_integral_estimate | <= rel_tol.

    Parameters
    ----------
    rel_tol:
        Relative error tolerance on consecutive integral mean values.
    """

    def __init__(self, rel_tol: FloatArgType):
        self.rel_tol = rel_tol

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
    max_evals:
        Maximum number of function evaluations.
    """

    def __init__(self, max_evals: IntArgType):
        self.max_evals = max_evals

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return bq_state.info.nevals >= self.max_evals
