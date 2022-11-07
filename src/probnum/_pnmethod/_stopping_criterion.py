"""Stopping criterion for a probabilistic numerical method."""

import abc
from typing import Any, Callable


class StoppingCriterion(abc.ABC):
    """Stopping criterion of a probabilistic numerical method.

    Checks whether quantities tracked by the probabilistic numerical
    method meet a desired terminal condition.

    Examples
    --------
    Stopping criteria support binary arithmetic, which makes them easy to combine.
    Take the following example, where we define a custom solver state.

    >>> import dataclasses
    >>> import numpy as np
    >>> from probnum import StoppingCriterion

    >>> @dataclasses.dataclass
    ... class SolverState:
    ...     iters = 50
    ...     atol = 1e-12
    ...     rtol = 1e-3

    >>> state = SolverState()

    Next we implement a few stopping criteria.

    >>> class MaxIterations(StoppingCriterion):
    ...     def __init__(self, maxiters):
    ...         self.maxiters = maxiters
    ...     def __call__(self, solver_state) -> bool:
    ...         return solver_state.iters >= self.maxiters

    >>> class AbsoluteResidualTolerance(StoppingCriterion):
    ...     def __init__(self, atol=1e-6):
    ...         self.atol = atol
    ...     def __call__(self, solver_state) -> bool:
    ...         return solver_state.atol < self.atol

    >>> class RelativeResidualTolerance(StoppingCriterion):
    ...     def __init__(self, rtol=1e-6):
    ...         self.rtol = rtol
    ...     def __call__(self, solver_state) -> bool:
    ...         return solver_state.rtol < self.rtol

    Now let's combine them by stopping when the solver has reached
    an absolute and relative tolerance, or a maximum number of iterations.

    >>> stopcrit = MaxIterations(maxiters=100) | (
    ...     AbsoluteResidualTolerance(atol=1e-6)
    ...     & RelativeResidualTolerance(rtol=1e-6)
    ... )

    >>> stopcrit(state)
    False

    Now let's modify the state such that the solver has reached
    a maximum number of iterations.

    >>> state.iters = 1000
    >>> stopcrit(state)
    True

    See Also
    --------
    LambdaStoppingCriterion : Stopping criterion defined via an anonymous function.
    ~probnum.linalg.solvers.stopping_criteria.LinearSolverStoppingCriterion : Stopping
        criterion of a probabilistic linear solver.
    ~probnum.filtsmooth.optim.FiltSmoothStoppingCriterion : Stopping criterion of
        filters and smoothers.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        """Check whether tracked quantities meet a desired terminal condition."""
        raise NotImplementedError

    def __and__(self, other):
        return LambdaStoppingCriterion(
            stopcrit=lambda *args, **kwargs: self(*args, **kwargs)
            and other(*args, **kwargs)
        )

    def __or__(self, other):
        return LambdaStoppingCriterion(
            stopcrit=lambda *args, **kwargs: self(*args, **kwargs)
            or other(*args, **kwargs)
        )

    def __invert__(self):
        return LambdaStoppingCriterion(
            stopcrit=lambda *args, **kwargs: not self(*args, **kwargs)
        )


class LambdaStoppingCriterion(StoppingCriterion):
    """Define a stopping criterion via an anonymous function.

    Defines a stopping criterion from a lambda function. This allows
    quick definition of stopping criteria for prototyping.

    Parameters
    ----------
    stopcrit
        Callable returning whether to stop or not.

    Examples
    --------
    >>> from probnum import LambdaStoppingCriterion
    >>> stopcrit = LambdaStoppingCriterion(lambda iters: iters >= 100)
    >>> stopcrit(101)
    True
    """

    def __init__(self, stopcrit: Callable[[Any], bool]) -> None:
        self._stopcrit = stopcrit

    def __call__(self, *args, **kwargs) -> bool:
        return self._stopcrit(*args, **kwargs)
