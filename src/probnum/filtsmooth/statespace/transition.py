"""Markov transition rules: continuous and discrete."""

import abc


__all__ = ["MarkovTransition"]


class MarkovTransition(abc.ABC):
    """
    Interface for Markov transition rules in discrete or continuous time.
    """

    def __call__(self, arr_or_rv):
        """Depending on the input, either call self.forward() or self.condition()"""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, arr, start, stop):
        raise NotImplementedError

    @abc.abstractmethod
    def condition(self, rv, start, stop):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, locations, size):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dimension(self):
        raise NotImplementedError
