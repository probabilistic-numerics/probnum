"""Discrete interventions."""

from typing import Callable, Union

import probnum  # pylint: disable="unused-import"
from probnum.diffeq.callbacks import _callback


class DiscreteCallback(_callback.ODESolverCallback):
    """Handle discrete events in an ODE solver.

    A discrete event can be any event for which it is possible to write down a condition
    that evaluates to `True` or `False`. If a condition evaluates to `True`, the current
    state can be modified/replaced.
    """

    # New init because condition() types are more specific.
    def __init__(
        self,
        replace: Callable[
            ["probnum.diffeq.ODESolverState"], "probnum.diffeq.ODESolverState"
        ],
        condition: Callable[["probnum.diffeq.ODESolverState"], Union[bool]],
    ):
        super().__init__(replace=replace, condition=condition)

    def __call__(
        self, state: "probnum.diffeq.ODESolverState"
    ) -> "probnum.diffeq.ODESolverState":
        if self.condition(state):
            state = self.replace(state)
        return state
