"""
Gaussian filtering.
Provides "filter()" method for Kalman-like filters
(KF, EKF, UKF).
"""

from abc import ABC, abstractmethod

import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth import bayesianfilter, LinearSDEModel, LTISDEModel



class GaussFiltSmooth(bayesianfilter.BayesianFilter, ABC):
    """
    """
    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
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
    def update(self, time, randvar, data, **kwargs):
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


class GaussianSmoother(GaussFiltSmooth):
    """
    Builds on top of a GaussianFilter. Passes things like predict()
    down to the filter and has its own smooth() methods instead of the
    filter() method.

    Out of the many types of smoothing, the current functionality is
    restricted to computing smoothing moments on the "mesh", i.e. the
    locations of the data.
    """
    def __init__(self, gaussfilt):
        """ """
        assert issubclass(type(gaussfilt), GaussianFilter)
        if issubclass(type(gaussfilt.dynamicmodel), LinearSDEModel):
            if not issubclass(type(gaussfilt.dynamicmodel), LTISDEModel):
                raise TypeError("Only LTI continuous models supported.")
        self.gaussfilt = gaussfilt

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        return self.gaussfilt.predict(start, stop, randvar, **kwargs)

    def update(self, time, randvar, data, **kwargs):
        """ """
        return self.gaussfilt.update(time, randvar, data, **kwargs)

    def smoothingstep(self, dist_from, predicted, currdist, crosscov):
        """ """
        ms1, ps1 = currdist.mean(), currdist.cov()
        mk, pk = dist_from.mean(), dist_from.cov()
        mk1, pk1 = predicted.mean(), predicted.cov()
        if np.isscalar(mk1) and np.isscalar(pk1):
            mk1, pk1 = mk1 * np.ones(1), pk1 * np.eye(1)
        newmean = mk + crosscov @ np.linalg.solve(pk1, ms1 - mk1)
        firstsolve = crosscov @ np.linalg.solve(pk1, ps1 - pk1)
        newcov = pk + (crosscov @ np.linalg.solve(pk1, firstsolve.T)).T
        return RandomVariable(distribution=Normal(newmean, newcov))

    @property
    def dynamicmodel(self):
        """ """
        return self.gaussfilt.dynamicmodel

    @property
    def measurementmodel(self):
        """ """
        return self.gaussfilt.measurementmodel

    @property
    def initialdistribution(self):
        """ """
        return self.gaussfilt.initialdistribution

    def smoother_stream(self, datastream, times, **kwargs):
        """
        First some filtering, then backwards in time some smoothing.
        """
        means, covs = self.gaussfilt.filter_stream(datastream, times, **kwargs)
        smoothed_means, smoothed_covs = self.smoothen_filteroutput(means, covs, times)
        return smoothed_means, smoothed_covs

    def smoothen_filteroutput(self, _means, _covs, times, **kwargs):
        """
        """
        means, covs = _means.copy(), _covs.copy()
        currdist = RandomVariable(distribution=Normal(means[-1], covs[-1]))
        for idx in reversed(range(1, len(times))):
            dist_from = RandomVariable(distribution=Normal(means[idx-1],
                                                           covs[idx-1]))
            predicted, crosscov = self.predict(times[idx-1], times[idx], dist_from, **kwargs)
            currdist = self.smoothingstep(dist_from, predicted, currdist, crosscov)
            means[idx-1], covs[idx-1] = currdist.mean(), currdist.cov()
        return means, covs

    def smoother(self, dataset, times, **kwargs):
        """
        """

        def set_as_stream(time):
            return dataset[times[1:] == time][0]

        return self.smoother_stream(set_as_stream, times, **kwargs)

class GaussianFilter(GaussFiltSmooth):
    """
    Maintains "abstractness" of predict() and update()
    of BayesianFilter class but adds a filter()
    method.
    Kalman-like filters (KF, EKF, UKF) inherit from
    GaussianFilter instead of BayesianFilter
    to leverage the filter() method.


    Note
    ----
    The subclasses are forced to implement the abstractmethods in
    GaussFiltSmooth. Not sure whether this is clean design, but for
    now it works.
    """

    def filter_stream(self, datastream, times, **kwargs):
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
            predicted, __ = self.predict(times[idx - 1], times[idx],
                                     currdist, **kwargs)
            data = datastream(times[idx], **kwargs)
            currdist, __, __, __ = self.update(times[idx], predicted, data, **kwargs)
            means[idx], covars[idx] = currdist.mean(), currdist.cov()
        return means, covars

    def filter(self, dataset, times, **kwargs):
        """
        """

        def set_as_stream(time):
            return dataset[times[1:] == time][0]

        return self.filter_stream(set_as_stream, times, **kwargs)
