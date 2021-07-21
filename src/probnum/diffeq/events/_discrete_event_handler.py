"""Discrete interventions."""

import numpy as np

from probnum.diffeq.events import _event_handler


class DiscreteEventHandler(_event_handler.EventHandler):
    """Handle discrete events in an ODE solver.

    A discrete event can be a time-stamp that must be included in the locations. It can
    also be any event for which it is possible to write down a condition that evaluates
    to `True` or `False`. If a condition evaluates to `True`, the current state can be
    modified.
    """

    def __init__(self, time_stamps, condition=None, modify=None):

        if condition is None:
            condition = lambda x: False
        super().__init__(condition=condition)

        if modify is None:
            modify = lambda x: x
        self.modify = modify

        # Initialize time stamps.
        self.time_stamps = time_stamps
        self.current_time_stamp_index = 0
        self.next_time_stamp = time_stamps[self.current_time_stamp_index]

    def interfere_dt(self, t, dt):
        """Check whether the next time-point is supposed to be stopped at."""

        if t + dt > self.next_time_stamp:
            new_dt = self.next_time_stamp - t
            self.advance_current_time_stamp()
            return new_dt
        return dt

    def advance_current_time_stamp(self):
        self.current_time_stamp_index += 1
        if self.current_time_stamp_index >= len(self.time_stamps):
            self.next_time_stamp = np.inf
        else:
            self.next_time_stamp = self.time_stamps[self.current_time_stamp_index]

    def intervene_state(self, state):
        if self.condition(state):
            new_rv = self.modify(state.rv)

            # We copy the state with the modified random variable.
            # Error estimate und reference state are copied on purpose.
            # By the time intervene_state is called, only these two values
            # will decide whether the step will be accepted or not.
            # The modification must not influence this decision.
            state = type(state)(
                rv=new_rv,
                t=state.t,
                error_estimate=state.error_estimate,
                reference_state=state.reference_state,
            )
        return state
