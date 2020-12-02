"""Coordinate changes in state space models."""

import abc

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.special  # for vectorised factorial


class Preconditioner(abc.ABC):
    """Coordinate change transformations as preconditioners in state space models.

    For some models, this makes the filtering and smoothing steps more
    numerically stable.
    """

    @abc.abstractmethod
    def __call__(self, step) -> np.ndarray:
        # if more than step is needed, add them into the signature in the future
        raise NotImplementedError

    @cached_property
    def inverse(self) -> "Preconditioner":
        raise NotImplementedError


class NordsieckLikeCoordinates(Preconditioner):
    """Nordsieck-like coordinates.

    Similar to Nordsieck coordinates (which store the Taylor
    coefficients instead of the derivatives), but better for ODE
    filtering and smoothing. Used in IBM.
    """

    def __init__(self, powers, scales, spatialdim):
        # Clean way of assembling these coordinates cheaply,
        # because the powers and scales of the inverse
        # are better read off than inverted
        self.powers = powers
        self.scales = scales
        self.spatialdim = spatialdim

    @classmethod
    def from_order(cls, order, spatialdim):
        # used to conveniently initialise in the beginning
        powers = np.arange(order, -1, -1)
        scales = scipy.special.factorial(powers)
        return cls(powers=powers + 0.5, scales=scales, spatialdim=spatialdim)

    def __call__(self, step):
        scaling_vector = step ** self.powers / self.scales
        return np.kron(np.eye(self.spatialdim), np.diag(scaling_vector))

    @cached_property
    def inverse(self) -> "NordsieckLikeCoordinates":
        return NordsieckLikeCoordinates(
            powers=-self.powers, scales=1.0 / self.scales, spatialdim=self.spatialdim
        )
