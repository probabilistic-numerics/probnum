"""
Abstract ODESolver class. Interface for Runge-Kutta, ODEFilter.
"""

from abc import ABC, abstractmethod


class ODESolver(ABC):
    """
    Interface for ODESolver.
    """

    @abstractmethod
    def solve(self, ivp, minstep, maxstep, **kwargs):
        """
        Every ODE solver has a solve() method.
        Optional: callback function. Allows  e.g. printing variables
        at runtime.
        """
        raise NotImplementedError
