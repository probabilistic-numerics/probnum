"""Event handler interface."""

import abc
from typing import Callable, Union

import probnum  # pylint: disable="unused-import"

__all__ = ["ODESolverCallback"]


class ODESolverCallback(abc.ABC):
    """Interface for pure callback-type events."""

    def __init__(
        self,
        replace: Callable[
            ["probnum.diffeq.ODESolverState"], "probnum.diffeq.ODESolverState"
        ],
        condition: Callable[["probnum.diffeq.ODESolverState"], Union[float, bool]],
    ):
        self.condition = condition
        self.replace = replace

    @abc.abstractmethod
    def __call__(
        self, state: "probnum.diffeq.ODESolverState"
    ) -> "probnum.diffeq.ODESolverState":
        """Modify a state whenever a condition dictates doing so."""
        raise NotImplementedError
