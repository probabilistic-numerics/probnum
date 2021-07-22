"""Event handler interface."""

import abc
from typing import Callable, Tuple, Union

from probnum.diffeq import stepsize
from probnum.typing import FloatArgType

__all__ = ["EventHandler", "CallbackEventHandler"]


PerformStepFunctionType = Callable[
    ["_odesolver.ODESolver.State", FloatArgType, stepsize.StepRule],
    Tuple["_odesolver.ODESolver.State", float],
]
"""Implementation of a perform_full_step() function. Interface according to ODESolver."""


class EventHandler(abc.ABC):
    """Interface for event handlers."""

    @abc.abstractmethod
    def __call__(
        self, perform_step_function: PerformStepFunctionType
    ) -> PerformStepFunctionType:
        """Wrap a perform_full_step implementation into a version that knows event
        handling functionality."""
        raise NotImplementedError


class CallbackEventHandler(EventHandler):
    """Interface for pure callback-type events."""

    def __init__(
        self,
        modify: Callable[["_odesolver.ODESolver.State"], "_odesolver.ODESolver.State"],
        condition: Callable[["_odesolver.ODESolver.State"], Union[float, bool]],
    ):
        self.condition = condition
        self.modify = modify

    def __call__(
        self, perform_step_function: PerformStepFunctionType
    ) -> PerformStepFunctionType:
        def new_perform_step_function(
            state: "_odesolver.ODESolver.State",
            dt: FloatArgType,
            steprule: stepsize.StepRule,
        ) -> Tuple["_odesolver.ODESolver.State", float]:
            """Modify the state after each performed step."""

            new_state, dt = perform_step_function(state, dt, steprule)
            new_state = self.modify_state(new_state)
            return new_state, dt

        return new_perform_step_function

    @abc.abstractmethod
    def modify_state(
        self, state: "_odesolver.ODESolver.State"
    ) -> "_odesolver.ODESolver.State":
        """Modify a state whenever a condition dictates doing so."""
        raise NotImplementedError


#
# One might implement a continuous callback as follows:
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
#     def modify_state(self, state):
#         if self.condition(state.x) < 0:
#             composed = lambda t: self.condition(state.dense_output(t).mean)
#             new_t, new_x = self.root_finding_algorithm(composed)
#
#             # new solver.State object including dense output!
#             new_state = solver.new_state(t=new_t, x=new_x)
#             return self.modify(new_state)
#         return state
