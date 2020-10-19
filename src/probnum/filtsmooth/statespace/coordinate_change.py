"""Coordinate changes in state space models."""

import abc


class CoordinateChange(abc.ABC):
    """Coordinate changes in transitions."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def from_linop(self, linop_forward, linop_inverse=None):
        raise NotImplementedError

