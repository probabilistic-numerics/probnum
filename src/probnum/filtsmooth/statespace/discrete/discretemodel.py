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
    #
    #
    # @abstractmethod
    # def sample(self, time, state, **kwargs):
    #     """
    #     Samples x_{t} ~ p(x_{t} | x_{s})
    #     as a function of t and x_s (plus additional parameters).
    #     """
    #     raise NotImplementedError
    #
    # def pdf(self, loc, time, state, **kwargs):
    #     """
    #     Evaluates pdf of p(x_t | x_s).
    #     Required for particle filtering and should be
    #     possible to implement for every reasonable model.
    #     """
    #     raise NotImplementedError("PDF not implemented.")
    #
    # @property
    # @abstractmethod
    # def ndim(self):
    #     raise NotImplementedError
