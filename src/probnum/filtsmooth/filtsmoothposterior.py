"""Abstract Base Class for posteriors over states after applying filtering/smoothing."""
from abc import ABC, abstractmethod


class FiltSmoothPosterior(ABC):
    """Posterior Distribution over States after Filtering/Smoothing."""

    @abstractmethod
    def __call__(self, location):
        """Evaluate the time-continuous posterior for a given location.

        Parameters
        ----------
        location : float
            Location, or time, at which to evaluate the posterior.

        Returns
        -------
        rv : `RandomVariable`
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Length of the discrete-time solution.

        Corresponds to the number of filtering/smoothing steps
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Return the corresponding index/slice of the discrete-time solution."""
        raise NotImplementedError

    def sample(self, locations=None, size=()):
        """Draw samples from the filtering/smoothing posterior.

        If nothing is specified, a single sample is drawn (supported on self.locations).
        If locations are specified, the samples are drawn on those locations.
        If size is specified, more than a single sample is drawn.

        Parameters
        ----------
        locations : array_like, optional
            Locations on which the samples are wanted. Default is none, which implies that
            self.location is used.
        size : int or tuple of ints, optional
            Indicates how many samples are drawn. Default is an empty tuple, in which case
            a single sample is returned.

        Returns
        -------
        numpy.ndarray
            Drawn samples. If size has shape (A1, ..., Z1), locations have shape (L,),
            and the state space model has shape (A2, ..., Z2), the output has
            shape (A1, ..., Z1, L, A2, ..., Z2).
            For example: size=4, len(locations)=4, dim=3 gives shape (4, 4, 3).
        """
        raise NotImplementedError("Sampling not implemented.")
