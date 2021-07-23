"""Time-stamps."""

from typing import Iterable

import numpy as np

from probnum.diffeq.events import _event_handler


class _TimeStopper:
    """Make the ODE solver stop at specified time-points."""

    def __init__(self, locations: Iterable):
        self._locations = iter(locations)
        self._next_location = next(self._locations)

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
