"""Time-stamps."""

import numpy as np

from probnum.diffeq.events import _event_handler


class TimeStampStopper(_event_handler.EventHandler):
    """Make the ODE solver stop at specified time-stamps."""

    def __init__(self, time_stamps):
        self._time_stamps = iter(time_stamps)
        self._next_time_stamp = next(self._time_stamps)

    def __call__(self, perform_step_function):
        """Wrap an ODE solver step() implementation into a step() implementation that
        knows events."""

        def new_perform_step_function(state, dt, steprule):
            """ODE solver steps that check for event handling."""

            new_dt = self.adjust_dt_to_next_time_stamp(t=state.t, dt=dt)
            new_state, dt = perform_step_function(state, new_dt, steprule)
            return new_state, dt

        return new_perform_step_function

    def adjust_dt_to_next_time_stamp(self, t, dt):
        """Check whether the next time-point is supposed to be stopped at."""

        if t + dt > self._next_time_stamp:
            dt = self._next_time_stamp - t
            self._advance_current_time_stamp()
        return dt

    def _advance_current_time_stamp(self):
        try:
            self._next_time_stamp = next(self._time_stamps)
        except StopIteration:
            self._next_time_stamp = np.inf
