"""Interface for transition test functions."""

import abc


class InterfaceTestTransition(abc.ABC):
    @abc.abstractmethod
    def test_forward_rv(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def test_forward_realization(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def test_backward_rv(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def test_backward_realization(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def test_input_dim(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def test_output_dim(self, *args, **kwargs):
        raise NotImplementedError
