"""Stopping criterion for a (probabilistic) numerical method."""

import abc


class StoppingCriterion(abc.ABC):
    r"""Stopping criterion of a (probabilistic) numerical method.

    Checks whether quantities tracked by the (probabilistic) numerical
    method meet a desired terminal condition.

    See Also
    --------
    LinearSolverStopCrit : Stopping criterion of a probabilistic linear solver.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        """Check whether tracked quantities meet a desired terminal condition."""
        raise NotImplementedError
