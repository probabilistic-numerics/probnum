"""
Stopping criterion: stop after relative tolerance is reached.


"""

from abc import ABC, abstractmethod

import numpy as np

from probnum.optim import Objective, Eval


class StoppingCriterion(ABC):
    """
    Interface for stopping criterion as used in Optimiser class.
    """

    @abstractmethod
    def fulfilled(self, curriter, lastiter, *args, **kwargs):
        """
        """
        raise NotImplementedError

    @abstractmethod
    def create_unfulfilled(self, iter, *args, **kwargs):
        """
        For initialisation of optimisers, we need an opportunity
        to create an iteration which does NOT fulfill the stopping
        criterion, not even by accident.
        """
        raise NotImplementedError


class NormOfGradient(StoppingCriterion):
    """
    """

    def __init__(self, tol):
        """
        """
        assert tol >= 0, "Please enter nonnegative atol"
        self._tol = tol

    def fulfilled(self, curriter, lastiter, *args, **kwargs):
        """
        lastiter: only included because of required signature, not used here.
        """
        if curriter.dfx is None:
            raise AttributeError("This class requires gradients.")
        if np.linalg.norm(curriter.dfx) < self._tol:
            return True
        else:
            return False

    def create_unfulfilled(self, curriter, *args, **kwargs):
        """
        This should have no input?????
        """
        not_fulfilling = np.inf
        output = Eval(None, None, not_fulfilling, None)
        assert self.fulfilled(output, None) is False
        return output


class DiffOfFctValues(StoppingCriterion):
    """
    """

    def __init__(self, tol):
        """
        """
        assert tol >= 0, "Please enter nonnegative rtol"
        self._tol = tol

    def fulfilled(self, curriter, lastiter, *args, **kwargs):
        """
        """
        if curriter.fx is None or lastiter.fx is None:
            raise AttributeError("This class requires function values.")
        if np.linalg.norm(curriter.fx - lastiter.fx) < self._tol:
            return True
        else:
            return False

    def create_unfulfilled(self, curriter, *args, **kwargs):
        """
        """
        output = Eval(None, curriter.fx + 10 * self._tol, None, None)
        assert self.fulfilled(curriter, output) is False
        return output
