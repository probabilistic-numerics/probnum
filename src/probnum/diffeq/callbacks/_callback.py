"""Event handler interface."""

import abc
from typing import Callable, Tuple, Union

from probnum.diffeq import stepsize
from probnum.typing import FloatArgType

__all__ = ["ODESolverCallback"]


class ODESolverCallback(abc.ABC):
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
