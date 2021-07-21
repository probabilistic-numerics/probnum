"""Event handler interface."""

import abc
from typing import Callable, Union

from probnum import randvars


class EventHandler(abc.ABC):
    """Interface for event handlers."""

    def __init__(
        self, condition: Callable[[randvars.RandomVariable], Union[bool, float]]
    ):
        self.condition = condition

    def __call__(self, perform_step_function):
        """Wrap an ODE solver step() implementation into a step() implementation that
        knows events."""

        def new_perform_step_function(start, stop, current_rv):
            """ODE solver steps that check for event handling."""
            new_dt = self.interfere_dt(t=start, dt=stop - start)
            new_state = perform_step_function(start, start + new_dt, current_rv)
            new_state = self.intervene_state(new_state)
            return new_state

        return new_perform_step_function

    def interfere_dt(self, t, dt):
        """Check whether the next time-point is supposed to be stopped at."""
        # Default behaviour: do nothing to the step.
        # Overwritten by discrete interventions (which handle event time-stamps).
        # Any other intervention (i.e. continuous interventions, do not do anything here).
        return dt

    @abc.abstractmethod
    def intervene_state(self, state):
        """Do nothing by default."""
        raise NotImplementedError


#
# class ContinuousInterventions(EventHandler):
#     """Whenever a condition : X -> R is zero, do something.
#
#     Examples
#     --------
#     >>> event = lambda t, x: x - 30
#     >>> intervene = lambda x: x + 2
#     >>> cont_events = ContinuousInterventions(condition=event, intervene=intervene)
#     """
#
#
#     def __init__(self, condition: ContinuousEvent, intervene, root_finding_method="bisect", **root_finding_kwargs):
#         self.condition = condition  # ContinuousEvent
#         self.intervene = intervene  # callable
#
#         def scipy_root(f):
#             return scipy.optimize.root_scalar(f, method=root_finding_method, **root_finding_kwargs)
#         self.root_finding_algorithm = scipy_root
#
#
#     def intervene_state(self, state, solver):
#         if self.condition(state.x) < 0:
#             composed = lambda t: self.condition(state.dense_output(t))
#             new_t, new_x = self.root_finding_algorithm(composed)
#
#             # new solver.State object including dense output!
#             new_state = solver.new_state(t=new_t, x=new_x)
#             return self.affect(new_state)
#         return state
