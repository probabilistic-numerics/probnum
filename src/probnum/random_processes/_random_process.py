"""
Random Processes.
"""

from abc import ABC, abstractmethod
from typing import Callable

from probnum.type import ShapeArgType

from ..random_variables import RandomVariable


class RandomProcess(ABC):
    """Random process."""

    @abstractmethod
    def __call__(self, location) -> RandomVariable:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        location
        """
        raise NotImplementedError

    def mean(self, location: DomainType) -> ValueType:
        raise NotImplementedError

    def std(self, location: DomainType) -> ValueType:
        raise NotImplementedError

    def var(self, location: DomainType) -> ValueType:
        raise NotImplementedError

    def cov(self, location1: DomainType, location2: DomainType) -> ValueType:
        raise NotImplementedError

    def sample(self, size: ShapeArgType = (), location: DomainType):
        """
        Sample paths from the random process.

        Parameters
        ----------
        location
            Evaluation location of the sample paths of the process.
        size
            Size of the sample.
        """
        if location is None:
            return lambda loc: self.sample(location=loc, size=size)
        else:
            return self._sample_at_locations(location=location, size=size)

    def _sample_at_locations(self, location: DomainType, size: ShapeArgType = ()):
        raise NotImplementedError
