"""Approximate information operators."""

import abc

from probnum.diffeq.odefiltsmooth import information_operators


class ApproximationStrategy(abc.ABC):
    """Turn an information operator into a tractable measurement model."""

    def __call__(
        self, information_operator: information_operators.InformationOperator
    ) -> information_operators.ApproximateInformationOperator:
        raise NotImplementedError
