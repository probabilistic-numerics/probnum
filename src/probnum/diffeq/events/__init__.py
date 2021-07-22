"""Event handling in ProbNum ODE solvers."""

from ._discrete_event_handler import DiscreteEventHandler
from ._event_handler import CallbackEventHandler, EventHandler
from ._time_stopper import TimeStopper

__all__ = [
    "EventHandler",
    "CallbackEventHandler",
    "DiscreteEventHandler",
    "TimeStopper",
]


EventHandler.__module__ = "probnum.diffeq.events"
CallbackEventHandler.__module__ = "probnum.diffeq.events"
DiscreteEventHandler.__module__ = "probnum.diffeq.events"
TimeStopper.__module__ = "probnum.diffeq.events"
