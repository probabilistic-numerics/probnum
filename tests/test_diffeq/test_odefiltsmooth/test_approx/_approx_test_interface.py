"""Interface for approximation strategies."""

import abc


class ApproximationStrategyTest(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError
