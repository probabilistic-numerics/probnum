"""Approximate information operators."""

import abc

from probnum.diffeq.odefilter import information_operators


class ApproximationStrategy(abc.ABC):
    """Interface for approximation strategies.

    Turn an information operator into an approximate information operator that converts
    into a :class:`ODEFilter` compatible :class:`Transition`.
    """

    def __call__(
        self, information_operator: information_operators.InformationOperator
    ) -> information_operators.ApproximateInformationOperator:
        """Derive a tractable approximation of an information operator."""
        raise NotImplementedError
