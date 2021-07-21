"""Discrete interventions."""

import numpy as np

from probnum.diffeq.events import _event_handler


class DiscreteEventHandler(_event_handler.EventHandler):
    """Whenever a condition: X -> {True, False} is True, do something.

    This object also handles a set of time-stamps that the solver shall cross.

    Examples
    --------
    >>> t = [1., 2., 3.]
    >>> events = DiscreteEventHandler(time_stamps=t)
    >>> solver = ODESolver(event_handler=events)
    >>> solver.solve()  # output contains [1., 2., 3.]

    >>> condition = lambda t, x: x==2.
    >>> modify = lambda x: x + 2
    >>> disc_events = DiscreteEventHandler(time_stamps=t, condition=condition, modify=modify)
    >>> solver = ODESolver(event_handler=disc_events)
    >>> solver.solve()
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
            state = type(state)(
                rv=new_rv, t=state.t, error_estimate=None, reference_state=None
            )
        return state
