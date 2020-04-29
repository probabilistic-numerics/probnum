"""
Transition densities for graphical models (state space models).
"""

from abc import ABC, abstractmethod
import numpy as np
from probnum.prob import RandomVariable, Normal


class Transition(ABC):
    """
    Markov transition densities.

    Representation of time-discrete probabilistic models given by
    a transition density :math:`p(x_i | x_{i-1})`. For example,
    this can be a transformation of a Gaussian
    :math:`p(x_i | x_{i-1}) = N(2x_i, 0.1)`.

    Used for e.g. graphical models and for time-discrete filtering
    and smoothing.

    Notes
    -----
    The main difference between :meth:`forward` and :meth:`condition`
    is that the former should always be available but the latter not.
    """

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


class GaussianTransition(Transition):
    """
    Transition models with additive Gaussian noise.

    That is, models of the form

    .. math:: p(x_i | x_{i-1}) = N(f(t_{i-1}), x_{i-1}), Q(t_{i-1}),

    are implemented. Jacobian is w.r.t. x.

    Examples
    --------
    >>> from probnum.prob.randomprocess import GaussianTransition
    >>> lgt = GaussianTransition(transfun=(lambda t, x: np.sin(x)), covfun=(lambda t: 0.1))
    >>> forw = lgt.forward(0, 1, value=0.2)
    >>> print(forw.mean(), forw.cov())
    0.19866933079506122 0.1
    """
    def __init__(self, transfun, covfun, jacobfun=None):
        """ """
        self._transfun = transfun
        self._covfun = covfun
        self._jacobfun = jacobfun

    # parameter "stop" is only here bc. of the general signature.
    def forward(self, start, stop, value, **kwargs):
        """
        """
        mean = self._transfun(start, value)
        cov = self._covfun(start)
        return RandomVariable(distribution=Normal(mean, cov))

    def condition(self, start, stop, randvar, **kwargs):
        """
        Only works if f=f(t, x) is linear in x.

        See :class:`LinearGaussianTransition`.
        """
        raise NotImplementedError

    def jacobfun(self, t, x):
        """ """
        return self._jacobfun(t, x)


class LinearGaussianTransition(GaussianTransition):
    """
    Linear Gaussian transitions.

    That is, the dynamic transition function :math:`f` is of the form
    :math:`f(t, x) = F(t) x`. This enables conditioning the
    distribution on a previous Gaussian distribution.

    Examples
    --------
    >>> from probnum.prob.randomprocess import LinearGaussianTransition
    >>> lgt = LinearGaussianTransition(lintransfun=(lambda t: 2), covfun=(lambda t: 0.1))
    >>> forw = lgt.forward(0, 1, value=0.2)
    >>> cond = lgt.condition(1, 2, randvar=forw)
    >>> print(forw.mean(), forw.cov())
    0.4 0.1
    >>> print(cond.mean(), cond.cov())
    0.8 0.4
    """
    def __init__(self, lintransfun, covfun):
        self._lintransfun = lintransfun
        super().__init__(transfun=(lambda t, x: np.dot(lintransfun(t), x)),
                         covfun=covfun, jacobfun=lintransfun)

    def condition(self, start, stop, randvar, **kwargs):
        """ """
        if not issubclass(type(randvar.distribution), Normal):
            raise ValueError("Input distribution must be a Normal.")
        oldmean, oldcov = randvar.mean(), randvar.cov()
        lintrans = self._lintransfun(start)
        if np.isscalar(oldmean):
            newmean = lintrans * oldmean
            newcov = lintrans**2 * oldcov
        else:
            newmean = lintrans @ oldmean
            newcov = lintrans @ oldcov @ lintrans.T
        return RandomVariable(distribution=Normal(newmean, newcov))

    def lintransfun(self, t):
        """ """
        return self._lintransfun(t)
