"""Interface for tests of information operators."""

import abc

from probnum.problems.zoo import diffeq as diffeq_zoo

import pytest


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


class ODEInformationOperatorTest(InformationOperatorTest):
    @abc.abstractmethod
    def test_incorporate_ode(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_ode_has_been_incorporated(self):
        raise NotImplementedError

    @pytest.fixture
    def fitzhughnagumo(self):
        return diffeq_zoo.fitzhughnagumo()
