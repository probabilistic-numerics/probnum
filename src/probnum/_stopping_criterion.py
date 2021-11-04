"""Stopping criterion for a (probabilistic) numerical method."""


class StoppingCriterion:
    r"""Stopping criterion of a (probabilistic) numerical method.

    Checks whether quantities tracked by the (probabilistic) numerical
    method meet a desired terminal condition.

    Examples
    --------
    Stopping criteria support boolean arithmetic, which makes them easy to combine.

    >
    >
    >

    See Also
    --------
    LinearSolverStopCrit : Stopping criterion of a probabilistic linear solver.
    """

    def __call__(self, *args, **kwargs) -> bool:
        """Check whether tracked quantities meet a desired terminal condition."""
        raise NotImplementedError

    def __and__(self, other):
        combined_stopcrit = StoppingCriterion()
        combined_stopcrit.__call__ = lambda *args, **kwargs: self(
            *args, **kwargs
        ) and other(*args, **kwargs)
        return combined_stopcrit

    def __or__(self, other):
        combined_stopcrit = StoppingCriterion()
        combined_stopcrit.__call__ = lambda *args, **kwargs: self(
            *args, **kwargs
        ) or other(*args, **kwargs)
        return combined_stopcrit

    def __invert__(self):
        inverted_stopcrit = StoppingCriterion()
        inverted_stopcrit.__call__ = lambda *args, **kwargs: ~self(*args, **kwargs)
        return inverted_stopcrit
