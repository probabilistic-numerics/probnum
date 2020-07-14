"""
Gaussian filtering.
"""
import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth.statespace import *
from probnum.filtsmooth.bayesfiltsmooth import *


def gaussfiltsmooth(dataset, times, method, **kwargs):
    """
    Apply Gaussian filtering and smoothing to a data set.

    Parameters
    ----------
    dataset : array_like, shape (N, M)
        Some data set that is to be filtered and smoothed.
    times : array_like, shape (N,)
        Temporal locations of the data.
    method : GaussFiltSmooth object
        Method that shall be used for filtering and smoothing,
        for instance `Kalman`.

    Returns
    -------
    array_like
        means of the filter output
    array_like
        covariances of the filter output
    """
    means, covs = gaussfilter(dataset, times, method, **kwargs)
    smoothed_rv = RandomVariable(distribution=Normal(means[-1], covs[-1]))
    for idx in reversed(range(1, len(times))):
        unsmoothed_rv = RandomVariable(
            distribution=Normal(means[idx-1], covs[idx-1]))
        pred_rv, ccov = method.predict(
            times[idx-1], times[idx], unsmoothed_rv, **kwargs)
        smoothed_rv = method.smoother_step(unsmoothed_rv, pred_rv, smoothed_rv, ccov)
        means[idx-1], covs[idx-1] = smoothed_rv.mean(), smoothed_rv.cov()
    return means, covs

def gaussfilter(dataset, times, method, **kwargs):
    """
    Apply Gaussian filtering to a data set.

    Parameters
    ----------
    dataset : array_like, shape (N, M)
        Some data set that is to be filtered and smoothed.
    times : array_like, shape (N,)
        Temporal locations of the data.
    method : GaussFiltSmooth object
        Method that shall be used for filtering, for instance `Kalman`.

    Returns
    -------
    array_like
        means of the filter output
    array_like
        covariances of the filter output
    """
    filtrv = method.initialrandomvariable
    means = [filtrv.mean()]
    covs = [filtrv.cov()]
    for idx in range(1, len(times)):
        filtrv = method.filter_step(
            start=times[idx-1], stop=times[idx],
            randvar=filtrv, data=dataset[idx-1],  **kwargs)
        means.append(filtrv.mean())
        covs.append(filtrv.cov())
    return np.array(means), np.array(covs)


class GaussFiltSmooth(BayesFiltSmooth, ABC):
    """
    Gaussian filtering and smoothing.

    Implements `filter_step` and `smoother_step`.
    """

    def __init__(self, dynamod, measmod, initrv):
        """
        """
        if not issubclass(type(initrv.distribution), Normal):
            raise ValueError("Gaussian filters need initial random "
                             "variables with Normal distribution.")
        # check for Gaussianity here.
        super().__init__(dynamod, measmod, initrv)


    def filter_step(self, start, stop, randvar, data, **kwargs):
        """
        A single filter step.

        Parameters
        ----------
        start :
        stop :
        randvar :
        data :

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        """
        predrv, _ = self.predict(start, stop, randvar, **kwargs)
        filtrv, _, _, _ = self.update(stop, predrv, data)
        return filtrv

    def smoother_step(self, unsmoothed_rv, pred_rv, smoothed_rv, crosscov):
        """
        A single smoother step.

        Parameters
        ----------
        unsmoothed_rv :
        pred_rv :
        smoothed_rv :
        crosscov :
            cross-covariance between unsmoothed_rv and pred_rv as
            returned by predict().
        """
        initmean, initcov = unsmoothed_rv.mean(), unsmoothed_rv.cov()
        predmean, predcov = pred_rv.mean(), pred_rv.cov()
        currmean, currcov = smoothed_rv.mean(), smoothed_rv.cov()
        if np.isscalar(predmean) and np.isscalar(predcov):
            predmean = predmean * np.ones(1)
            predcov = predcov * np.eye(1)
        res = currmean - predmean
        newmean = initmean + crosscov @ np.linalg.solve(predcov, res)
        firstsolve = crosscov @ np.linalg.solve(predcov, currcov - predcov)
        secondsolve = (crosscov @ np.linalg.solve(predcov, firstsolve.T))
        newcov = initcov + secondsolve.T
        return RandomVariable(distribution=Normal(newmean, newcov))

    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, **kwargs):
        raise NotImplementedError
