"""Stopping criterion for a (probabilistic) numerical method."""


class StoppingCriterion:
    r"""Stopping criterion of a (probabilistic) numerical method.

    Checks whether quantities tracked by the (probabilistic) numerical
    method meet a desired terminal condition.

    Examples
    --------
    Stopping criteria support boolean arithmetic, which makes them easy to combine.
    Take the following example, where we define a custom solver state.

    >>> import dataclasses
    >>> import numpy as np
    >>> import probnum as pn

    >>> @dataclasses.dataclass
    ... class SolverState:
    ...     iters = 50
    ...     atol = 1e-3
    ...     rtol = 1e-3

    >>> state = SolverState()

    Next we implement a few custom stopping criteria.

    >>> class MaxIterations(pn.StoppingCriterion):
    ...     def __init__(self, maxiters):
    ...         self.maxiters = maxiters
    ...     def __call__(self, solver_state) -> bool:
    ...         return solver_state.iters >= self.maxiters

    >>> class AbsoluteResidualTolerance(pn.StoppingCriterion):
    ...     def __init__(self, atol=1e-6):
    ...         self.atol = atol
    ...     def __call__(self, solver_state) -> bool:
    ...         return solver_state.atol < self.atol

    >>> class RelativeResidualTolerance(pn.StoppingCriterion):
    ...     def __init__(self, rtol=1e-6):
    ...         self.rtol = rtol
    ...     def __call__(self, solver_state) -> bool:
    ...         return solver_state.rtol < self.rtol

    Now let's combine them by stopping when the solver has reached an absolute and relative tolerance, or a maximum number of iterations.

    >>> stopcrit = MaxIterations(maxiters=100) or (AbsoluteResidualTolerance(atol=1e-6, rtol=1e-5) and RelativeResidualTolerance(rtol=1e-6))
    >>> stopcrit(state)
    False

    Now let's modify the state such that the solver has reached a maximum number of iterations.

    >>> state.iters = 1000
    >>> stopcrit(state)
    True

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
