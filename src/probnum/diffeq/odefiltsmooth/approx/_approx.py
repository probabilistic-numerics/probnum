"""Approximate information operators."""

import abc

from probnum import statespace
from probnum.diffeq.odefiltsmooth import information_operators


class ODEInformationApproximationStrategy(abc.ABC):
    def __call__(
        self, ode_info: information_operators.ODEInformation
    ) -> statespace.DiscreteGaussian:
        raise NotImplementedError
