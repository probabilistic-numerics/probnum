"""
Transition densities for graphical models (state space models).
"""

from abc import ABC, abstractmethod
from probnum.prob import RandomVariable


class Transition(ABC):
    """
    Interface for (Markov) transition densities.

    Representation of time-discrete probabilistic models given by
    a transition density :math:`p(x_i | x_{i-1})`. For example,
    this can be a transformation of a Gaussian
    :math:`p(x_i | x_{i-1}) = N(2x_i, 0.1)`.

    Used for e.g. graphical models and for time-discrete filtering
    and smoothing. This object also serves as an interface to
    stochastic differential equations, because that is the usual way of
    formulating Markov processes in continuous time.

    Notes
    -----
    The main difference between :meth:`forward` and :meth:`condition`
    is that the former should always be available but the latter not.
    """
    def __init__(self, support=None):
        """ """
        if support is None:
            support = []
        self._support = support

    @property
    def support(self):
        """
        Return the support of the transition density.

        If the transition density represents a discrete process,
        this is a list of values, e.g. [1, 2, 3, 4, 5].
        If the transition density represents a continuous temporal
        process (through an SDE), this is a tuple of values (t0, tmax).
        If the transition density represents a discrete spatial
        process (a Markov random field),
        this is a graph (nodes, weights).
        """
        return self._support

    @support.setter
    def support(self, support):
        """
        Replace the support with a new support.

        If the transition density represents a discrete process,
        this function expectes a list of values, e.g. [1, 2, 3, 4, 5].
        If the transition density represents a continuous temporal
        process (through an SDE), this function expects a tuple
        of values (t0, tmax).
        If the transition density represents a discrete spatial
        process (a Markov random field), this function expects
        a graph (dict of nodes and weights).
        """
        # todo: do some type checking here
        self._support = support

    @abstractmethod
    def forward(self, start, stop, value, **kwargs):
        """
        Computes next iteration of the transition model.

        This function computes the distribution of the next state
        :math:`x_i` given the value :math:`z`of the last state
        :math:`x_{i-1}`,

        .. math:: x_i \\sim p(x_i | x_{i-1}=z).

        Parameters
        ----------
        start : float
            Time :math:`t_{i-1}` of the previous state :math:`x_{i-1}`.
            In many implementations of discrete-time transition
            densities this will be ignored. Sometimes the transition
            also depends on the time, in which case this parameter
            is important.
        stop : float
            Time :math:`t_i` of the next state :math:`x_i`.
            In most implementations of discrete-time transition
            densities this will be ignored. In the time-continuous
            case (see :class:`SDE`) this variable is important.
        value : unspecified
            Value :math:`z` that the random process can attend.
            In probnum, this is usually an ``array_like`` or a float,
            but can technically be anything.
        kwargs : optional
            Optional arguments to be passed down to implementations of
            this method in subclasses.

        Returns
        -------
        RandomVariable
            Random variable :math:`x_i` with distribution
            :math:`p(x_i | x_{i-1}=z)`.
        """
        return NotImplementedError

    @abstractmethod
    def condition(self, start, stop, randvar, **kwargs):
        """
        Conditions conditional distribution given previous distribution.

        If the distribution of a previous state
        :math:`x_{i-1} \\sim p(x_{i-1})` has an explicit
        parameterisation (e.g. a mean and covariance), this function
        returns the parameterisation of of :math:`p(x_i | x_{i-1})`.

        For example, if the transition is a linear Gaussian, this
        function takes a Gaussian random variable with mean :math:`m`
        and covariance :math:`C` and returns a Gaussian random variable
        with transformed mean and covariance.

        Parameters
        ----------
        start : float
            Time :math:`t_{i-1}` of the previous state :math:`x_{i-1}`.
            In many implementations of discrete-time transition
            densities this will be ignored. Sometimes the transition
            also depends on the time, in which case this parameter
            is important.
        stop : float
            Time :math:`t_i` of the next state :math:`x_i`.
            In most implementations of discrete-time transition
            densities this will be ignored. In the time-continuous
            case (see :class:`SDE`) this variable is important.
        randvar : RandomVariable
            Law of :math:`x_{i-1}` as a RandomVariable object.
        kwargs : optional
            Optional arguments to be passed down to implementations of
            this method in subclasses.

        Returns
        -------
        RandomVariable
            Law of :math:`p(x_i | x_{i-1})` as a RandomVariable object.
        """
        return NotImplementedError

