"""Discrete interventions."""

from typing import Callable, Union

from probnum.diffeq.events import _event_handler


class DiscreteEventHandler(_event_handler.CallbackEventHandler):
    """Handle discrete events in an ODE solver.

    A discrete event can be a time-stamp that must be included in the locations. It can
    also be any event for which it is possible to write down a condition that evaluates
    to `True` or `False`. If a condition evaluates to `True`, the current state can be
    modified.
    """

    # New init because condition() types are more specific.
    def __init__(
        self,
        modify: Callable[["_odesolver.ODESolver.State"], "_odesolver.ODESolver.State"],
        condition: Callable[["_odesolver.ODESolver.State"], Union[bool]],
    ):
        super().__init__(modify=modify, condition=condition)

    def modify_state(
        self, state: "_odesolver.ODESolver.State"
    ) -> "_odesolver.ODESolver.State":
        if self.condition(state):
            state = self.modify(state)
        return state
