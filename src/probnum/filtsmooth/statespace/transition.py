"""Markov transition rules: continuous and discrete."""

import abc
from typing import Union, Dict

from probnum.random_variables import RandomVariable
from probnum.typing import FloatArgType


__all__ = ["Transition"]


class Transition(abc.ABC):
    """
    Markov transition rules in discrete or continuous time.

    In continuous time, this is a Markov process and described by a
    stochastic differential equation (SDE)

    .. math:: d x_t = f(t, x_t) d t + d w_t

    driven by a Wiener process :math:`w`. In discrete time, it is defined by
    a transformation

    .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t).

    Sometimes, these can be equivalent. For example: mild solutions to
    linear, time-invariant SDEs have an equivalent, discretised form that can
    be written as a transformation.

    See Also
    --------
    :class:`ContinuousModel`
        Continuously indexed transitions (SDEs)
    :class:`DiscreteModel`
        Discretely indexed transitions (transformations)
    """

    def __call__(
        self, arr_or_rv, start: FloatArgType = None, stop: FloatArgType = None, **kwargs
    ) -> (RandomVariable, Dict):
        """
        Apply the transition.

        The input is either interpreted as a random variable or as a realization.
        Accordingly, the respective methods are called: :meth:`transition_realization` or :meth:`transition_rv`.
        """
        if isinstance(arr_or_rv, RandomVariable):
            return self.transition_rv(rv=arr_or_rv, start=start, stop=stop, **kwargs)
        return self.transition_realization(
            real=arr_or_rv, start=start, stop=stop, **kwargs
        )

    @abc.abstractmethod
    def transition_realization(self, real, start, stop, **kwargs):
        """
        Apply transition to a realization of a random variable from time :math:`t` to time :math:`t+\\Delta t`.

        For random variable :math:`x_t`, it returns the random variable defined by

        .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t = r) .

        This is different to :meth:`transition_rv` which computes the parametrization
        of :math:`x_{t + \\Delta t}` based on the parametrization of :math:`x_t`.

        Parameters
        ----------
        real : array_like
            Realization of the random variable.
        start : float
            Starting point :math:`t`.
        stop : float
            End point :math:`t + \\Delta t`.

        Returns
        -------
        RandomVariable
            Random variable, describing the state at time :math:`t + \\Delta t`
            based on realization at time :math:`t`.
        dict
            Additional information in form of a dictionary,
            for instance the cross-covariance in the
            prediction step, access to which is useful in smoothing.

        See Also
        --------
        :meth:`transition_rv`
            Apply transition to a random variable.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(self, rv, start, stop, **kwargs):
        """
        Apply transition to a random variable from time :math:`t` to time :math:`t+\\Delta t`.

        For random variable :math:`x_t`, it returns the random variable defined by

        .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t) .

        This returns a random variable where the parametrization depends on the paramtrization of :math:`x_t`.
        This is different to :meth:`transition_rv` which computes the parametrization
        of :math:`x_{t + \\Delta t}` based on a realization of :math:`x_t`.


        Parameters
        ----------
        rv : array_like
            Realization of the random variable.
        start : float
            Starting point :math:`t`.
        stop : float
            End point :math:`t + \\Delta t`.

        Returns
        -------
        RandomVariable
            Random variable, describing the state at time :math:`t + \\Delta t`
            based on realization at time :math:`t`.
        dict
            Additional information in form of a dictionary,
            for instance the cross-covariance in the
            prediction step, access to which is useful in smoothing.

        See Also
        --------
        :meth:`transition_realization`
            Apply transition to a realization of a random variable.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dimension(self):
        """Dimension of the transition model."""
        raise NotImplementedError
