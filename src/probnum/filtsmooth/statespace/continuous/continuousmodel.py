"""
Continuous Markov models implicitly defined through SDEs,
dx(t) = f(t, x(t)) dt + l(t, x(t)) dB(t).
Possible input for StateSpace.dynmod or StateSpace.measmod.

Note
----
ContinuousModel only implements
StateSpaceComponent.sample(). It passes on the
responsibility of implementing chapmankolmogorov() to its
subclasses and creates further abstract methods drift(),
dispersion() as well as a hook for jacobian().

When implementing continuous models we think of
Gauss-Markov models (see linearsde.py)
but do our very best to not enforce Gaussianity.
In the future, this might be more general than Gauss-
Markov models.
"""

import abc

import numpy as np

from probnum.filtsmooth.statespace.transition import Transition

__all__ = ["ContinuousModel"]


class ContinuousModel(Transition):
    """
    Interface for time-continuous
    Markov models of the form
    dx = f(t, x) dt + l(t, x) dBt.

    In the language of dynamic models:
    x(t) : state process
    f(t, x(t)) : drift function
    l(t, x(t)) : dispersion matrix.
    B(t) : Brownian motion with const. diffusion matrix Q.
    """

    @abc.abstractmethod
    def transition_realization(self, real, start, stop, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(self, rv, start, stop, *args):
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

    @property
    @abc.abstractmethod
    def dimension(self):
        """
        Spatial dimension (utility attribute).
        """
        raise NotImplementedError
    #
    # def chapmankolmogorov(self, start, stop, step, randvar, **kwargs):
    #     """
    #     If available, this returns the closed form solution to the
    #     Chapman-Kolmogorov equations (CKEs).
    #
    #     Solutions to the CKEs are important in filtering.
    #
    #     Available for instance for linear SDEs.
    #     """
    #     raise NotImplementedError("Chap.-Kolg. not implemented.")
