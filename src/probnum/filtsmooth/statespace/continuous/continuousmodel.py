"""
Continuous Markov models implicitly defined through SDEs,
dx(t) = f(t, x(t)) dt + l(t, x(t)) dB(t).
"""

import abc

import numpy as np

from probnum.filtsmooth.statespace.transition import Transition

__all__ = ["ContinuousModel"]


class ContinuousModel(Transition):
    """
    Markov transition rules or continuous time.

    This is described by a stochastic differential equation (SDE),

    .. math:: d x_t = f(t, x_t) d t + d w_t

    driven by a Wiener process :math:`w`.

    Todo
    ----
    This should be initializable similarly to :math:`DiscreteGaussianModel`
    (where :meth:`transition_realization` and :meth:`transition_rv` simply raise Errors).
    """

    @abc.abstractmethod
    def transition_realization(self, real, start, stop, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(self, rv, start, stop, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dimension(self):
        raise NotImplementedError

    @abc.abstractmethod
    def drift(self, time, state, **kwargs):
        """
        Evaluates f = f(t, x(t)).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dispersion(self, time, state, **kwargs):
        """
        Evaluates l = l(t, x(t)).
        """
        raise NotImplementedError

    def jacobian(self, time, state, **kwargs):
        """
        Jacobian of drift w.r.t. state: d_x f(t, x(t))
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def diffusionmatrix(self):
        """
        Evaluates Q.
        In 1D, this is \\sigma^2.
        """
        raise NotImplementedError
