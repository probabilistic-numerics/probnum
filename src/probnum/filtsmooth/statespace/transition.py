"""Markov transition rules: continuous and discrete."""

import abc
from probnum.random_variables import RandomVariable

__all__ = ["MarkovTransition"]


class MarkovTransition(abc.ABC):
    """
    Interface for Markov transition rules in discrete or continuous time.
    """

    def __call__(self, arr_or_rv, *args):
        """Depending on the input, either call self.transition_array() or self.transition_rv()"""
        if isinstance(arr_or_rv, RandomVariable):
            return self.transition_rv(arr_or_rv, *args)
        return self.transition_array(arr_or_rv, *args)

    def sample(self, arr_or_rv, size=(), *args):
        """Sample from the transition."""
        return self.__call__(arr_or_rv, *args).sample(size=size)

    @abc.abstractmethod
    def transition_array(self, arr, start, stop, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(self, rv, start, stop, *args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dimension(self):
        raise NotImplementedError
