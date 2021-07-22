"""Discrete interventions."""

import numpy as np

from probnum.diffeq.events import _event_handler


class DiscreteEventHandler(_event_handler.EventCallback):
    """Handle discrete events in an ODE solver.

    A discrete event can be a time-stamp that must be included in the locations. It can
    also be any event for which it is possible to write down a condition that evaluates
    to `True` or `False`. If a condition evaluates to `True`, the current state can be
    modified.
    """

    def modify_state(self, state):
        if self.condition(state):
            state = self.modify(state)
        return state
