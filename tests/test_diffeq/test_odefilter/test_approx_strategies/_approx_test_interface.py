"""Interface for approximation strategies."""

import abc

from probnum.problems.zoo import diffeq as diffeq_zoo

import pytest


class ApproximationStrategyTest(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @pytest.fixture
    def fitzhughnagumo(self):
        return diffeq_zoo.fitzhughnagumo()
