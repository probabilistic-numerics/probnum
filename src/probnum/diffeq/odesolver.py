"""
Abstract ODESolver class. Interface for Runge-Kutta, ODEFilter.
"""

from abc import ABC, abstractmethod


class ODESolver(ABC):
    """
    Interface for ODESolver.
    """

    # solve() will be made non-abstract soon
    @abstractmethod
    def solve(self, ivp, minstep, maxstep, **kwargs):
        """
        Every ODE solver has a solve() method.
        Optional: callback function. Allows  e.g. printing variables
        at runtime.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, start, stop, current, **kwargs):
        """Every ODE solver needs a step() method that returns a new random variable and an error estimate"""
        raise NotImplementedError

    def pre_accepted_callback(self, time, current_guess, current_error):
        """Do this as soon as it is clear that the current guess is accepted, but before storing it. No return."""
        pass