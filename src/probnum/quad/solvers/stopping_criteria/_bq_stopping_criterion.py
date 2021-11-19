"""Base class for Bayesian quadrature stopping criteria."""

from probnum import StoppingCriterion
from probnum.quad.solvers.bq_state import BQState

# pylint: disable=too-few-public-methods, fixme
# pylint: disable=arguments-differ


class BQStoppingCriterion(StoppingCriterion):
    r"""Stopping criterion of a Bayesian quadrature method.

    Checks whether quantities tracked by the :class:`~probnum.quad.solvers.BQState` meet a desired terminal condition.

    See Also
    --------
    IntegralVarianceTolerance : Stop based on the variance of the integral estimator.
    RelativeMeanChange : Stop based on the absolute value of the integral variance.
    MaxNevals : Stop based on a maximum number of iterations.
    """

    def __call__(self, bq_state: BQState) -> bool:
        """Check whether tracked quantities meet a desired terminal condition.

        Parameters
        ----------
        bq_state:
            State of the BQ loop.
        """
        raise NotImplementedError
