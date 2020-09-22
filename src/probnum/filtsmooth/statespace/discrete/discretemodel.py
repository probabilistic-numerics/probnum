import abc

from probnum.filtsmooth.statespace.transition import Transition

__all__ = ["DiscreteModel"]


class DiscreteModel(Transition):
    """
    Abstract interface for state space model components.
    x(t_{i+1}) ~ p(x(t_{i+1}) | x(t_{i})).

    Nothing happens here except passing responsibilities
    of implementation down the subclasses.
    In the future, this might change so please subclass
    this object accordingly.
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
