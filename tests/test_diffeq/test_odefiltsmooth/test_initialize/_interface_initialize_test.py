"""Interface for python initialize tests."""

import abc


class InterfaceInitializationRoutineTest(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_is_exact(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_requires_jax(self):
        raise NotImplementedError
