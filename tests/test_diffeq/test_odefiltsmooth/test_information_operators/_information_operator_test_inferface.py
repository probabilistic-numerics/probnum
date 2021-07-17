"""Interface for tests of information operators."""

import abc


class InformationOperatorTest(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_jacobian(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_as_transition(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_as_ekf_component(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_incorporate_ode(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_ode_has_been_incorporated(self):
        raise NotImplementedError
