"""Base class for Bayesian quadrature stopping criteria."""

from probnum import StoppingCriterion
from probnum.quad.solvers.bq_state import BQState
from probnum.randvars import Normal


# Todo: update docstring
class BQStoppingCriterion(StoppingCriterion):
    r"""Stopping criterion of a Bayesian quadrature method.

    Checks whether quantities tracked by the :class:`~probnum.quad.solvers.BQState` meet a desired terminal condition.

    See Also
    --------
    IntegralVarianceTolerance : Stop based on the variance of the integral estimator.
    """

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        """Check whether tracked quantities meet a desired terminal condition.

        Parameters
        ----------
        integral_belief :
            Current Gaussian belief about the integral.
        bq_state:
            State of the BQ loop.
        """
        raise NotImplementedError
