"""Interface for tests of initialization routines of ODE filters."""

import abc


class InterfaceInitializationRoutineTest(abc.ABC):
    """Interface for tests of initialization routines."""

    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_is_exact(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_requires_jax(self):
        raise NotImplementedError
