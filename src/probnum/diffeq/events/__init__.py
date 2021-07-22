"""Event handling in ProbNum ODE solvers."""

from ._discrete_event_handler import DiscreteEventHandler
from ._event_handler import EventCallback, EventHandler
from ._time_stamp import TimeStampStopper

__all__ = ["EventHandler", "EventCallback", "DiscreteEventHandler", "TimeStampStopper"]


EventHandler.__module__ = "probnum.diffeq.events"
EventCallback.__module__ = "probnum.diffeq.events"
DiscreteEventHandler.__module__ = "probnum.diffeq.events"
TimeStampStopper.__module__ = "probnum.diffeq.events"
