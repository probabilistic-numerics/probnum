from abc import ABC, abstractmethod


__all__ = ["DiscreteModel"]


class DiscreteModel(ABC):
    """
    Abstract interface for state space model components.
    x(t_{i+1}) ~ p(x(t_{i+1}) | x(t_{i})).

    Nothing happens here except passing responsibilities
    of implementation down the subclasses.
    In the future, this might change so please subclass
    this object accordingly.
    """

    @abstractmethod
    def sample(self, time, state, **kwargs):
        """
        Samples x_{t} ~ p(x_{t} | x_{s})
        as a function of t and x_s (plus additional parameters).
        """
        raise NotImplementedError

    def pdf(self, loc, time, state, **kwargs):
        """
        Evaluates pdf of p(x_t | x_s).
        Required for particle filtering and should be
        possible to implement for every reasonable model.
        """
        raise NotImplementedError("PDF not implemented.")

    @property
    @abstractmethod
    def ndim(self):
        raise NotImplementedError
