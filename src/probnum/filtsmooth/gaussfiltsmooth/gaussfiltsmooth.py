"""
Gaussian filtering.
Provides "filter()" method for Kalman-like filters
(KF, EKF, UKF).
"""

from abc import ABC, abstractmethod

import numpy as np

from probnum.filtsmooth import bayesianfilter


__all__ = ["GaussianFilter"]



class GaussianFilter(bayesianfilter.BayesianFilter, ABC):
    """
    Maintains "abstractness" of predict() and update()
    of BayesianFilter class but adds a filter()
    method.
    Kalman-like filters (KF, EKF, UKF) inherit from
    GaussianFilter instead of BayesianFilter
    to leverage the filter() method.
    """

    @abstractmethod
    def predict(self, start, stop, randvar, *args, **kwargs):
        """
        Overwrites superclass' abstract method with another
        abstract method. Same interface.

        Arguments
        ---------
        randvar : auxiliary.randomvariable.RandomVariable object
            usually a Gaussian
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, *args, **kwargs):
        """
        Overwrites superclass' abstract method with another
        abstract method. Same interface.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dynamicmodel(self):
        """
        Overwrites superclass' abstract method with another
        abstract method. Same interface.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def measurementmodel(self):
        """
        Overwrites superclass' abstract method with another
        abstract method. Same interface.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def initialdistribution(self):
        """
        Overwrites superclass' abstract method with another
        abstract method. Same interface.
        """
        raise NotImplementedError

    def filter_stream(self, datastream, times, *args, **kwargs):
        """
        Returns arrays of means and covariances.
        Assumes discrete measurement model.
        Able to handle both continuous and discrete dynamics.


        Arguments
        ---------
        datastream: callable, maps t to (d,) array.
            Data stream for filtering.
        times: np.ndarray, shape (ndata + 1, d)
            time steps of the observations.
        """
        ndim = self.dynamicmodel.ndim
        means = np.zeros((len(times), ndim))
        covars = np.zeros((len(times), ndim, ndim))
        means[0] = self.initialdistribution.mean()
        covars[0] = self.initialdistribution.cov()
        currdist = self.initialdistribution
        for idx in range(1, len(times)):
            predicted = self.predict(times[idx - 1], times[idx],
                                     currdist, *args, **kwargs)
            data = datastream(times[idx], *args, **kwargs)
            currdist, __, __, __ = self.update(times[idx], predicted, data, *args, **kwargs)
            means[idx], covars[idx] = currdist.mean(), currdist.cov()
        return means, covars

    def filter(self, dataset, times, *args, **kwargs):
        """
        """

        def set_as_stream(time):
            return dataset[times[1:] == time][0]

        return self.filter_stream(set_as_stream, times, *args, **kwargs)
