"""Abstract Base Class for posteriors over states after applying filtering/smoothing"""
from abc import ABC, abstractmethod


class FiltSmoothPosterior(ABC):
    """Posterior Distribution over States after Filtering/Smoothing"""

    @abstractmethod
    def __call__(self, location):
        """Evaluate the time-continuous posterior for a given location

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
        """Length of the discrete-time solution

        Corresponds to the number of filtering/smoothing steps
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Return the corresponding index/slice of the discrete-time solution"""
        raise NotImplementedError

    def sample(self, locations=None, size=None):
        """
        Draw samples from the filtering/smoothing posterior.


        If location is None, it returns N joint samples from the list of RVs;
        if location is one of the locations of the posterior,
        if not (i.e. if it is a scalar), it returns a sample from the specified location.

        If size is empty, it is a single sample. If not, multiple samples at once.
        """
        raise NotImplementedError("Sampling not implemented.")
