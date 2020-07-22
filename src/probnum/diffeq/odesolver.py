"""
Abstract ODESolver class. Interface for Runge-Kutta, ODEFilter.
"""

from abc import ABC, abstractmethod


class ODESolver(ABC):
    """
    Interface for ODESolver.
    """

    def __init__(self, steprule):
        """
        An ODESolver is an object governed by a stepsize rule.
        That is: constant steps or adaptive steps.
        """
        self.steprule = steprule

    @abstractmethod
    def solve(self, ivp, minstep, maxstep, **kwargs):
        """
        Every ODE solver has a solve() method.
        Optional: callback function. Allows  e.g. printing variables
        at runtime.
        """
        raise NotImplementedError
