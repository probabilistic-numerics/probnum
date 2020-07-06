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

    This is messy AF at the moment.
    """
    def __init__(self, gaussfilt):
        """ """
        # do check for GaussianFilter type here which
        # implies checks for Gaussianity
        super().__init__(gaussfilt)

    def smooth(self, data, times, **kwargs):
        """ """
        mns, cvs, tms = self.bayesfilt.filter(data, times, **kwargs)
        new_mns, new_cvs = self.smooth_filteroutput(mns, cvs, tms, **kwargs)
        return new_mns, new_cvs, tms

    def smooth_filteroutput(self, means, covs, times, **kwargs):
        """ """
        currdist = RandomVariable(distribution=Normal(means[-1], covs[-1]))
        for idx in reversed(range(1, len(times))):
            dist_from = RandomVariable(distribution=Normal(means[idx-1],
                                                           covs[idx-1]))
            pred, ccov = self.bayesfilt.predict(times[idx-1], times[idx],
                                                dist_from, **kwargs)
            currdist = self.smoother_step(dist_from, pred, currdist, ccov)
            means[idx-1], covs[idx-1] = currdist.mean(), currdist.cov()
        return means, covs

    def smoother_step(self, dist_from, predicted, currdist, crosscov):
        """
        """
        # todo: give variables better names
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
    """

    def __init__(self, dynamod, measmod, initrv):
        """
        """
        if not issubclass(type(initrv.distribution), Normal):
            raise ValueError("Gaussian filters need initial random "
                             "variables with Normal distribution.")
        # check for Gaussianity here.
        super().__init__(dynamod, measmod, initrv)

    def filter_set(self, dataset, times, **kwargs):
        """ """
        def set_as_stream(tm, **kwargs):
            return dataset[times[1:] == tm][0]

        return self.filter_stream(set_as_stream, times, **kwargs)

    def filter_stream(self, datastream, times, **kwargs):
        """ """
        means = [self.initialdistribution.mean()]
        covars = [self.initialdistribution.cov()]
        fitimes = [times[0]]
        currdist = self.initialdistribution
        for idx in range(1, len(times)):
            data = datastream(times[idx], **kwargs)
            currdist, mns, cvs, tms = self.filter_step(start=times[idx-1],
                                                       stop=times[idx],
                                                       randvar=currdist,
                                                       data=data, **kwargs)
            means.extend(mns)
            covars.extend(cvs)
            fitimes.extend(tms)
        return np.array(means), np.array(covars), np.array(fitimes)

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

    @abstractmethod
    def filter_step(self, start, stop, randvar, data, **kwargs):
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

    def filter_step(self, start, stop, randvar, data, **kwargs):
        """
        Filter step of continuous-discrete Gaussian filter.

        First a fixed number of prediction steps, then an update step.

        Parameters
        ----------
        start
        stop
        randvar
        data
        kwargs

        Returns
        -------

        """
        if "nsteps" in kwargs.keys():  # number of steps in between obs.
            nsteps = kwargs["nsteps"]
        else:
            nsteps = 1
        intermediate_step = float((stop - start) / nsteps)
        tm, currdist = start, randvar
        intermns, intercvs, intertms = [], [], []
        for jdx in range(nsteps):
            currdist, __ = self.predict(tm, tm + intermediate_step,
                                        currdist, **kwargs)
            tm = tm + intermediate_step
            intertms.append(tm)
            intermns.append(currdist.mean())
            intercvs.append(currdist.cov())
        currdist, __, __, __ = self.update(stop, currdist, data, **kwargs)
        intermns[-1] = currdist.mean()
        intercvs[-1] = currdist.cov()
        return currdist, intermns, intercvs, intertms

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

    def filter_step(self, start, stop, randvar, data, **kwargs):
        """
        Filter step of discrete-discrete Gaussian filter.

        First a prediction step, then an update step

        Parameters
        ----------
        start
        stop
        randvar
        data
        kwargs

        Returns
        -------
        RandomVariable
        list
        list
        list
        """
        pred, __ = self.predict(start, stop, randvar, **kwargs)
        currdist, __, __, __ = self.update(stop, pred, data, **kwargs)
        return currdist, [currdist.mean()], [currdist.cov()], [stop]

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
