"""Interface for event handler tests."""

import abc


class EventHandlerTest(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError
