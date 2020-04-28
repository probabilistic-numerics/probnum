"""
Gaussian filtering.
"""


import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth.statespace import *
from probnum.filtsmooth.bayesfiltsmooth import *


class GaussianSmoother(BayesSmoother):
    """
    Gaussian smoothing.

    Builds on top of GaussianFilter instances

    This is messy AF code ATM.
    """
    def __init__(self, gaussfilt):
        """ """
        # do check for GaussianFilter type here which
        # implies checks for Gaussianity
        super().__init__(gaussfilt)

    def smooth(self, data, times, **kwargs):
        """ """
        if isinstance(data, np.ndarray):
            return self.smooth_set(data, times, **kwargs)
        elif callable(data):
            return self.smooth_stream(data, times, **kwargs)
        else:
            errormsg = "`data` parameter is not of expected type."
            raise ValueError(errormsg)

    def smooth_set(self, dataset, times, **kwargs):
        """
        """

        def set_as_stream(time, **kwargs):
            return dataset[times[1:] == time][0]

        return self.smooth_stream(set_as_stream, times, **kwargs)

    def smooth_stream(self, datastream, times, **kwargs):
        """ """
        means, covs, fitimes = self.bayesfilt.filter_stream(datastream, times, **kwargs)
        smmeans, smcovs = self.smooth_filterout(means, covs, fitimes, **kwargs)
        return smmeans, smcovs, fitimes

    def smooth_filterout(self, means, covs, times, **kwargs):
        """ """
        currdist = RandomVariable(distribution=Normal(means[-1], covs[-1]))
        for idx in reversed(range(1, len(times))):
            dist_from = RandomVariable(distribution=Normal(means[idx-1],
                                                           covs[idx-1]))
            pred, ccov = self.bayesfilt.predict(times[idx-1], times[idx],
                                                dist_from, **kwargs)
            currdist = self.smoothing_step(dist_from, pred, currdist, ccov)
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
        res = currmean - predmean
        newmean = initmean + crosscov @ np.linalg.solve(predcov, res)
        firstsolve = crosscov @ np.linalg.solve(predcov, currcov - predcov)
        secondsolve = (crosscov @ np.linalg.solve(predcov, firstsolve.T))
        newcov = initcov + secondsolve.T
        return RandomVariable(distribution=Normal(newmean, newcov))


class GaussianFilter(BayesFilter, ABC):
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
        if not issubclass(type(initrv.distribution), Normal):
            raise ValueError("Gaussian filters need initial random "
                             "variables with Normal distribution.")
        # check for Gaussianity here.
        super().__init__(dynamod, measmod, initrv)

    # todo: filter() method.

    def filter_set(self, dataset, times, **kwargs):
        """ """
        def set_as_stream(tm, **kwargs):
            return dataset[times[1:] == tm][0]

        return self.filter_stream(set_as_stream, times, **kwargs)

    @abstractmethod
    def filter_stream(self, datastream, times, **kwargs):
        """
        Gaussian filters have straightforward filtering,
        so subclasses must implement a filter_stream() method.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
        """
        Makes superclass' method abstract because all Gaussian filters
        can/must do a prediction step followed by an update step.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, **kwargs):
        """ """
        raise NotImplementedError


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

        Asserts that dynamod is continuous (a linear SDE)
        and measmod is discrete (Gaussian).

        """
        if not issubclass(type(dynamod), ContinuousModel):
            raise ValueError("ContDiscGaussianFilter needs a "
                             "continuous dynamic model.")
        if not issubclass(type(measmod), DiscreteModel):
            raise ValueError("ContDiscGaussianFilter needs a "
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

    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
        """
        Makes superclass' method abstract because all Gaussian filters
        can/must do a prediction step followed by an update step.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, **kwargs):
        """ """
        raise NotImplementedError


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
            raise ValueError("DiscreteDiscreteGaussianFilter needs a "
                             "discrete dynamic model.")
        if not issubclass(type(measmod), DiscreteModel):
            raise ValueError("DiscreteDiscreteGaussianFilter needs a "
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

    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
        """
        Makes superclass' method abstract because all Gaussian filters
        can/must do a prediction step followed by an update step.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, **kwargs):
        """ """
        raise NotImplementedError
