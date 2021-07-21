"""Interface for event handler tests."""

import abc


class EventHandlerTest(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_interfere_dt(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_intervene_state(self):
        raise NotImplementedError
