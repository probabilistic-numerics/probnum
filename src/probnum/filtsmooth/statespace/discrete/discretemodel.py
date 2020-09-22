import abc

from probnum.filtsmooth.statespace.transition import Transition

__all__ = ["DiscreteModel"]


class DiscreteModel(Transition):
    """
    Transition models for discretely indexed processes.

    Transformations of the form

    .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t) .

    As such, compatible with Bayesian filtering and smoothing algorithms.

    See Also
    --------
    :class:`ContinuousModel`
        Transition models for continuously indexed processes.
    :class:`BayesFiltSmooth`
        Bayesian filtering and smoothing algorithms.
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
