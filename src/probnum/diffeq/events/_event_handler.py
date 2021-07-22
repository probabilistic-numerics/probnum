"""Event handler interface."""

import abc
from typing import Callable, Tuple, Union

from probnum import randvars
from probnum.diffeq import stepsize
from probnum.typing import FloatArgType

PerformStepFunctionType = Callable[
    ["_odesolver.ODESolver.State", FloatArgType, stepsize.StepRule],
    Tuple["_odesolver.ODESolver.State", FloatArgType],
]


class EventHandler(abc.ABC):
    """Interface for event handlers.

    Event handlers decorate perform_full_step implementations bei either interferring
    before or after the step. For instance, enforcing fixed time-stamps into the
    solution grid requires interferring before the step. Modifying the state whenever a
    condition is true, however, is done after the step.
    """

    @abc.abstractmethod
    def __call__(
        self, perform_step_function: PerformStepFunctionType
    ) -> PerformStepFunctionType:
        raise NotImplementedError


class EventCallback(EventHandler):
    """Interface for pure callback-type events."""

    def __init__(self, condition, modify):
        self.condition = condition
        self.modify = modify

    def __call__(
        self, perform_step_function: PerformStepFunctionType
    ) -> PerformStepFunctionType:
        """Wrap an ODE solver step() implementation into a step() implementation that
        knows events."""

        def new_perform_step_function(state, dt, steprule):
            """ODE solver steps that check for event handling."""

            new_state, dt = perform_step_function(state, dt, steprule)
            new_state = self.modify_state(new_state)
            return new_state, dt

        return new_perform_step_function

    @abc.abstractmethod
    def modify_state(self, state):
        """Modify a state whenever a condition dictates doing so."""
        raise NotImplementedError


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
