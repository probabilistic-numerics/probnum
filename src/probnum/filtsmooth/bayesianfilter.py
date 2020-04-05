"""
"""

from abc import ABC, abstractmethod

__all__ = ["BayesianFilter"]

class BayesianFilter(ABC):
    """
    Cp. Theorem 4.1 in 'Bayesian Filtering and Smoothing'.

    Note
    ----
    The BayesianFilter object does not have a filter()
    method, nor does it force the subclasses to have one.
    Some do, some don't; either way, the bottleneck is
    getting the predict() and update() steps right.
    This is what we provide.
    """

    @abstractmethod
    def predict(self, start, stop, randvar, *args, **kwargs):
        """
        Pretends value of model at "stop" from "randvar" at "start".
        randvar : auxiliary.randomvariable.RandomVariable object
            usually a Gaussian
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, *args, **kwargs):
        """
        """
        raise NotImplementedError
