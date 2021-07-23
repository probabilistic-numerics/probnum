"""Event handling in ProbNum ODE solvers."""

from ._discrete_event_handler import DiscreteEventHandler
from ._event_handler import CallbackEventHandler

__all__ = [
    "CallbackEventHandler",
    "DiscreteEventHandler",
]


CallbackEventHandler.__module__ = "probnum.diffeq.events"
DiscreteEventHandler.__module__ = "probnum.diffeq.events"
