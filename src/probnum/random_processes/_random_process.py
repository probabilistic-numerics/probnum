"""
Random Processes.
"""

from typing import Generic, TypeVar

from probnum.type import ShapeArgType

from ..random_variables import RandomVariable

_DomainType = TypeVar("DomainType")
_ValueType = TypeVar("ValueType")


class RandomProcess(Generic[_DomainType, _ValueType]):
    """Random process."""

    def __call__(self, input) -> RandomVariable:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        input
        """
        raise NotImplementedError

    def mean(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def std(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def var(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def cov(self, input1: _DomainType, input2: _DomainType) -> _ValueType:
        raise NotImplementedError

    def sample(self, input: _DomainType, size: ShapeArgType = ()):
        """
        Sample paths from the random process.

        Parameters
        ----------
        input
            Evaluation input of the sample paths of the process.
        size
            Size of the sample.
        """
        if input is None:
            return lambda loc: self.sample(input=loc, size=size)
        else:
            return self._sample_at_inputs(input=input, size=size)

    def _sample_at_inputs(self, input: _DomainType, size: ShapeArgType = ()):
        raise NotImplementedError
