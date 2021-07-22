"""Time-stamps."""

from typing import Iterable

import numpy as np

from probnum.diffeq.events import _event_handler


class TimeStopper(_event_handler.EventHandler):
    """Make the ODE solver stop at specified time-points."""

    def __init__(self, locations: Iterable):
        self._locations = iter(locations)
        self._next_location = next(self._locations)

    def __call__(self, perform_step_function: _event_handler.PerformStepFunctionType):
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

        if t + dt > self._next_location:
            dt = self._next_location - t
            self._advance_current_location()
        return dt

    def _advance_current_location(self):
        try:
            self._next_location = next(self._locations)
        except StopIteration:
            self._next_location = np.inf
