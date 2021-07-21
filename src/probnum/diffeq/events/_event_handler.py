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

    def __call__(self, attempt_step_function):
        """Wrap an ODE solver step() implementation into a step() implementation that
        knows events."""

        def new_attempt_step_function(state, dt):
            """ODE solver steps that check for event handling."""

            new_dt = self.interfere_dt(t=state.t, dt=dt)
            new_state = attempt_step_function(state, new_dt)
            new_state = self.intervene_state(new_state)
            return new_state

        return new_attempt_step_function

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
#     If a sign-change happens, use dense output of the posterior to find the root.
#     """
#
#     def __init__(self, condition: Callable, intervene, root_finding_method="bisect", **root_finding_kwargs):
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
