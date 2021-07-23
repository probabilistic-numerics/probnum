"""Event handler interface."""

import abc
from typing import Callable, Tuple, Union

from probnum.diffeq import stepsize
from probnum.typing import FloatArgType

__all__ = ["EventHandler", "CallbackEventHandler"]


PerformStepFunctionType = Callable[
    ["probnum.diffeq.ODESolver.State", FloatArgType, stepsize.StepRule],
    Tuple["probnum.diffeq.ODESolver.State", float],
]
"""Implementation of a perform_full_step() function. Interface according to ODESolver."""


class CallbackEventHandler:
    """Interface for pure callback-type events."""

    def __init__(
        self,
        modify: Callable[
            ["probnum.diffeq.ODESolver.State"], "probnum.diffeq.ODESolver.State"
        ],
        condition: Callable[["probnum.diffeq.ODESolver.State"], Union[float, bool]],
    ):
        self.condition = condition
        self.modify = modify

    @abc.abstractmethod
    def __call__(
        self, state: "probnum.diffeq.ODESolver.State"
    ) -> "probnum.diffeq.ODESolver.State":
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
#             composed = lambda t: self.condition(state.dense_output(t))
#             new_t, new_x = self.root_finding_algorithm(composed)
#
#             # new solver.State object including dense output!
#             new_state = solver.new_state(t=new_t, x=new_x)
#             return self.modify(new_state)
#         return state
