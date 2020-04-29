"""
Transition densities for graphical models (state space models).
"""

from abc import ABC, abstractmethod

__all__ = ["Transition"]


class Transition(ABC):
    """
    Explicit transition densities p(x_i | x_{i-1}).

    Used for observation models and for discrete Random Processes.
    """

    @abstractmethod
    def forward(self, start, stop, value, **kwargs):
        """
        Transitions from x to p(x_i | x_{i-1}=x).

        Returns RandomVariable object.
        """
        return NotImplementedError

    @abstractmethod
    def condition(self, start, stop, randvar, **kwargs):
        """
        Conditions p(x_i | x_{i-1}) on explicit distribution p(x_{i-1}).

        For example: linear transformation of a Gaussian.

        Returns RandomVariable object.
        """
        return NotImplementedError

