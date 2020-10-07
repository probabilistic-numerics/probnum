"""
Continuous Markov models implicitly defined through SDEs,
dx(t) = f(t, x(t)) dt + l(t, x(t)) dB(t).
"""

import abc

from probnum.filtsmooth.statespace.transition import Transition

__all__ = ["ContinuousModel"]


class ContinuousModel(Transition):
    """
    Markov transition rules in continuous time.

    Such a rule is described by a stochastic differential equation (SDE),

    .. math:: d x_t = f(t, x_t) d t + d w_t

    driven by a Wiener process :math:`w`.

    Todo
    ----
    This should be initializable similarly to :class:`DiscreteGaussianModel`
    (where :meth:`transition_realization` and :meth:`transition_rv` simply raise ``NotImplementedError``).
    This would change a bit of code, though. See Issue #219.
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
        raise NotImplementedError

    @abc.abstractmethod
    def dispersion(self, time, state, **kwargs):
        raise NotImplementedError

    def jacobian(self, time, state, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def diffusionmatrix(self):
        raise NotImplementedError
