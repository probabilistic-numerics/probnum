"""Approximate information operators."""

import abc

from probnum import statespace


class ODEInformationApproximationStrategy(abc.ABC):
    """Turn an information operator into a tractable measurement model."""

    def __call__(self, information_operator) -> statespace.DiscreteGaussian:
        raise NotImplementedError
