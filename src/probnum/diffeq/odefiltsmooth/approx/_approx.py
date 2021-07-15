"""Approximate information operators."""

import abc

from probnum.diffeq.odefiltsmooth import information_operators


class ApproximateODEInformation(abc.ABC):
    def __call__(
        self, ode_info: information_operators.ODEInformation
    ) -> statespace.DiscreteGaussian:
        raise NotImplementedError
