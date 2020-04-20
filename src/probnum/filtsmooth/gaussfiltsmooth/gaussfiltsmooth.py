"""
Gaussian filtering.
Provides "filter()" method for Kalman-like filters
(KF, EKF, UKF).
"""

from abc import ABC, abstractmethod

import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth.statespace import *

__all__ = ["GaussianFilter", "GaussianSmoother", "ContContGaussianFilter",
           "ContDiscGaussianFilter", "DiscDiscGaussianFilter"]


# _GaussFiltSmooth #####################################################

class _GaussFiltSmooth:
    """
    General properties of filters and smoothers.

    E.g. properties
        * dynamicmodel
        * measurementmodel
        * initialdistribution
    """
    def __init__(self, dynamod, measmod, initrv):
        """ """
        if not issubclass(type(initrv.distribution), Normal):
            raise TypeError("Gaussian filters need initial random "
                            "variables with Normal distribution.")
        self.dynamod = dynamod
        self.measmod = measmod
        self.initrv = initrv

    @property
    def dynamicmodel(self):
        return self.dynamod

    @property
    def measurementmodel(self):
        return self.measmod

    @property
    def initialrandomvariable(self):
        return self.initrv

    @property
    def initialdistribution(self):
        """ """
        return self.initrv.distribution


# Gaussian Smoother ####################################################

class GaussianSmoother(_GaussFiltSmooth):
    """
    Gaussian smoothing.

    Builds on top of GaussianFilter instances
    """
    def __init__(self, gaussfilt):
        """ """
        self.gaussfilt = gaussfilt
        super().__init__(dynamod=gaussfilt.dynamicmodel,
                         measmod=gaussfilt.measurementmodel,
                         initrv=gaussfilt.initialrandomvariable)

    def smooth(self, dataset, times, **kwargs):
        """
        """

        def set_as_stream(time, **kwargs):
            return dataset[times[1:] == time][0]

        return self.smooth_stream(set_as_stream, times, **kwargs)

    def smooth_stream(self, datastream, times, **kwargs):
        """ """
        means, covs, fitimes = self.gaussfilt.filter_stream(datastream, times,
                                                           **kwargs)
        smmeans, smcovs = self.smooth_filterout(means, covs, fitimes, **kwargs)
        return smmeans, smcovs, fitimes

    def smooth_filterout(self, means, covs, times, **kwargs):
        """ """
        currdist = RandomVariable(distribution=Normal(means[-1], covs[-1]))
        for idx in reversed(range(1, len(times))):
            dist_from = RandomVariable(distribution=Normal(means[idx-1],
                                                           covs[idx-1]))
            predicted, crosscov = self.gaussfilt.predict(times[idx-1], times[idx], dist_from, **kwargs)
            currdist = self.smoothing_step(dist_from, predicted, currdist, crosscov)
            means[idx-1], covs[idx-1] = currdist.mean(), currdist.cov()
        return means, covs

    def smoothing_step(self, dist_from, predicted, currdist, crosscov):
        """
        Needs some meaningful naming when this is all over with.
        """
        currmean, currcov = currdist.mean(), currdist.cov()
        initmean, initcov = dist_from.mean(), dist_from.cov()
        predmean, predcov = predicted.mean(), predicted.cov()
        if np.isscalar(predmean) and np.isscalar(predcov):
            predmean = predmean * np.ones(1)
            predcov = predcov * np.eye(1)
        # print(np.linalg.cond(predcov))
        newmean = initmean + crosscov @ np.linalg.solve(predcov, currmean - predmean)
        firstsolve = crosscov @ np.linalg.solve(predcov, currcov - predcov)
        newcov = initcov + (crosscov @ np.linalg.solve(predcov, firstsolve.T)).T
        # gain = crosscov @ np.linalg.inv(predcov)
        # newmean = initmean + gain @ (currmean - predmean)
        # newcov = initcov + crosscov @ gain @ (currcov - predcov) @ gain.T
        return RandomVariable(distribution=Normal(newmean, newcov))


# Gaussian Filter ######################################################

class GaussianFilter(_GaussFiltSmooth, ABC):
    """
    Abstract interface for Gaussian filters.

    Has abstract methods:
        * :meth:`filter`         implemented by _**GaussianFilter, ...
        * :meth:`filter_stream`  implemented by _**GaussianFilter, ...
        * :meth:`predict`        implemented by Kalman, ExtendedKalman, ...
        * :meth:`update`         implemented by Kalman, ExtendedKalman, ...

    Hence, "proper" subclasses must do both: subclass one of
    {_ContinuousContinuousGaussianFilter, ...} and provide :meth:`predict` and
    :meth:`update` methods. It is recommended to duplicate the factory pattern
    from this class at all subclasses.
    """

    def __init__(self, dynamod, measmod, initrv):
        """
        """
        super().__init__(dynamod, measmod, initrv)

    def filter(self, dataset, times, **kwargs):
        """ """

        def set_as_stream(tm, **kwargs):
            return dataset[times[1:] == tm][0]

        return self.filter_stream(set_as_stream, times, **kwargs)

    @abstractmethod
    def filter_stream(self, datastream, times, **kwargs):
        """ """
        raise NotImplementedError

    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
        """ """
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, **kwargs):
        """ """
        raise NotImplementedError


class ContContGaussianFilter:
    """
    Implements filter() for cont.-cont. state space models.

    If you decide to implement it, feel free to subclass accordingly.
    """
    def __init__(self, dynamod, measmod, initrv):
        """ """
        raise NotImplementedError("Continuous/Continuous filtering and "
                                  "smoothing not supported.")


class ContDiscGaussianFilter(GaussianFilter):
    """
    Incomplete implementation of GaussianFilter for  continuous-discrete
    models.

    Misses predict() and update() which are provided by the type of
    Gaussian filter employed: KF, EKF, UKF, etc..

    Implements filter() and filter_stream() for cont.-disc. state space
    models.
    """
    def __init__(self, dynamod, measmod, initrv):
        """
        Cannot be created without subclassing and providing predict()
        and update().

        Asserts that dynamod is continuous and measmod is discrete.
        """
        if not issubclass(type(dynamod), ContinuousModel):
            raise TypeError("ContinuousDiscreteGaussianFilter needs a "
                            "continuous dynamic model.")
        if not issubclass(type(measmod), DiscreteModel):
            raise TypeError("ContinuousDiscreteGaussianFilter needs a "
                            "discrete measurement model.")

        super().__init__(dynamod, measmod, initrv)

    def filter_stream(self, datastream, times, **kwargs):
        """
        The zero-th element of 'times' is the time
        of initial distribution.
        """
        if "nsteps" in kwargs.keys():
            nsteps = kwargs["nsteps"]
        else:
            nsteps = 1
        filtertimes = [times[0]]
        means = [self.initialdistribution.mean()]
        covars = [self.initialdistribution.cov()]
        currdist = self.initialdistribution
        for idx in range(1, len(times)):
            intermediate_step = float((times[idx] - times[idx-1]) / nsteps)
            tm = times[idx - 1]
            for jdx in range(nsteps):
                currdist, __ = self.predict(tm, tm + intermediate_step,
                                            currdist, **kwargs)
                tm = tm + intermediate_step
                filtertimes.append(tm)
                means.append(currdist.mean())
                covars.append(currdist.cov())
            data = datastream(times[idx], **kwargs)
            currdist, __, __, __ = self.update(times[idx], currdist,
                                               data, **kwargs)
            means[-1] = currdist.mean()
            covars[-1] = currdist.cov()
        return np.array(means), np.array(covars), np.array(filtertimes)


class DiscDiscGaussianFilter(GaussianFilter):
    """
    Incomplete implementation of GaussianFilter for discrete models.

    Misses predict() and update() which are provided by the type of
    Gaussian filter employed: KF, EKF, UKF, etc..

    Implements filter() and filter_stream() for disc.-disc. state space
    models. This is the plain-vanilla filter.
    """
    def __init__(self, dynamod, measmod, initrv):
        """ """
        if not issubclass(type(dynamod), DiscreteModel):
            raise TypeError("DiscreteDiscreteGaussianFilter needs a "
                            "discrete dynamic model.")
        if not issubclass(type(measmod), DiscreteModel):
            raise TypeError("DiscreteDiscreteGaussianFilter needs a "
                            "discrete measurement model.")

        super().__init__(dynamod, measmod, initrv)

    def filter_stream(self, datastream, times, **kwargs):
        """ """
        means = [self.initialdistribution.mean()]
        covars = [self.initialdistribution.cov()]
        currdist = self.initialdistribution
        for idx in range(1, len(times)):
            pred, __ = self.predict(times[idx-1], times[idx],
                                         currdist, **kwargs)
            data = datastream(times[idx], **kwargs)
            currdist, __, __, __ = self.update(times[idx], pred,
                                               data, **kwargs)
            means.append(currdist.mean())
            covars.append(currdist.cov())
        return np.array(means), np.array(covars), times
