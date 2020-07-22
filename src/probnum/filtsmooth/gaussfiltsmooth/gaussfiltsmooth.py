"""
Gaussian filtering.
"""
import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth.bayesfiltsmooth import *


class GaussFiltSmooth(BayesFiltSmooth, ABC):
    """
    Interface for Gaussian filtering and smoothing.
    """

    def __init__(self, dynamod, measmod, initrv):
        """        Check that the initial distribution is Gaussian.        """
        if not issubclass(type(initrv.distribution), Normal):
            raise ValueError(
                "Gaussian filters/smoothers need initial "
                "random variables with Normal distribution."
            )
        super().__init__(dynamod, measmod, initrv)

    def filtsmooth(self, dataset, times, **kwargs):
        """
        Apply Gaussian filtering and smoothing to a data set.

        Parameters
        ----------
        dataset : array_like, shape (N, M)
            Data set that is filtered.
        times : array_like, shape (N,)
            Temporal locations of the data points.
        kwargs : ???

        Returns
        -------
        ndarray, shape (N, M)
            Means of the smoothed output
        ndarray, shape (N, M, M)
            Covariances of the smoothed output.
        """
        dataset, times = np.asarray(dataset), np.asarray(times)
        filtered_means, filtered_covs = self.filter(dataset, times, **kwargs)
        smoothed_means, smoothed_covs = self.smooth(
            filtered_means, filtered_covs, times, **kwargs
        )
        return smoothed_means, smoothed_covs

    def filter(self, dataset, times, **kwargs):
        """
        Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        dataset : array_like, shape (N, M)
            Data set that is filtered.
        times : array_like, shape (N,)
            Temporal locations of the data points.
        kwargs : ???

        Returns
        -------
        ndarray, shape (N, M)
            Means of the filtered output
        ndarray, shape (N, M, M)
            Covariances of the filtered output.
        """
        dataset, times = np.asarray(dataset), np.asarray(times)
        filtrv = self.initialrandomvariable
        means = [filtrv.mean()]
        covs = [filtrv.cov()]
        for idx in range(1, len(times)):
            filtrv = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                randvar=filtrv,
                data=dataset[idx - 1],
                **kwargs
            )
            means.append(filtrv.mean())
            covs.append(filtrv.cov())
        return np.array(means), np.array(covs)

    def smooth(self, filtmeans, filtcovs, times, **kwargs):
        """
        Apply Gaussian smoothing to a set of filtered means and covariances.

        Parameters
        ----------
        filtmeans : array_like, shape (N, M)
            Means of the filter solution.
        filtcovs : array_like, shape (N, M, M)
            Covariances of the filter solution.
        times : array_like, shape (N,)
            Temporal locations of the filter solution (i.e. locations of
            the data points---the data points are not needed anymore)
        kwargs : ???

        Returns
        -------
        ndarray, shape (N, M)
            Means of the smoothed filter solution
        ndarray, shape (N, M, M)
            Covariances of the smoothed filter solution

        """
        means, covs = np.zeros(filtmeans.shape), np.zeros(filtcovs.shape)
        means[-1], covs[-1] = filtmeans[-1], filtcovs[-1]
        smoothed_rv = RandomVariable(distribution=Normal(filtmeans[-1], filtcovs[-1]))
        for idx in reversed(range(1, len(times))):
            unsmoothed_rv = RandomVariable(
                distribution=Normal(filtmeans[idx - 1], filtcovs[idx - 1])
            )
            pred_rv, ccov = self.predict(
                times[idx - 1], times[idx], unsmoothed_rv, **kwargs
            )
            smoothed_rv = self.smooth_step(unsmoothed_rv, pred_rv, smoothed_rv, ccov)
            means[idx - 1], covs[idx - 1] = smoothed_rv.mean(), smoothed_rv.cov()
        return means, covs

    def filter_step(self, start, stop, randvar, data, **kwargs):
        """
        A single filter step.

        Consists of a prediction step (t -> t+1) and an update step (at t+1).

        Parameters
        ----------
        start : float
            Predict FROM this time point.
        stop : float
            Predict TO this time point.
        randvar : RandomVariable
            Predict based on this random variable. For instance, this can be the result of a previous call to filter_step.
        data : array_like
            Compute the update based on this data.

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        """
        data = np.asarray(data)
        predrv, _ = self.predict(start, stop, randvar, **kwargs)
        filtrv, _, _, _ = self.update(stop, predrv, data)
        return filtrv

    def smooth_step(self, unsmoothed_rv, pred_rv, smoothed_rv, crosscov):
        """
        A single smoother step.

        Consists of predicting from the filtering distribution at time t
        to time t+1 and then updating based on the discrepancy to the
        smoothing solution at time t+1.

        Parameters
        ----------
        unsmoothed_rv : RandomVariable
            Filtering distribution at time t.
        pred_rv : RandomVariable
            Prediction at time t+1 of the filtering distribution at time t.
        smoothed_rv : RandomVariable
            Smoothing distribution at time t+1.
        crosscov : array_like
            Cross-covariance between unsmoothed_rv and pred_rv as
            returned by predict().
        """
        crosscov = np.asarray(crosscov)
        initmean, initcov = unsmoothed_rv.mean(), unsmoothed_rv.cov()
        predmean, predcov = pred_rv.mean(), pred_rv.cov()
        currmean, currcov = smoothed_rv.mean(), smoothed_rv.cov()
        if np.isscalar(predmean) and np.isscalar(predcov):
            predmean = predmean * np.ones(1)
            predcov = predcov * np.eye(1)
        res = currmean - predmean
        newmean = initmean + crosscov @ np.linalg.solve(predcov, res)
        firstsolve = crosscov @ np.linalg.solve(predcov, currcov - predcov)
        secondsolve = crosscov @ np.linalg.solve(predcov, firstsolve.T)
        newcov = initcov + secondsolve.T
        return RandomVariable(distribution=Normal(newmean, newcov))

    @abstractmethod
    def predict(self, start, stop, randvar, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, time, randvar, data, **kwargs):
        raise NotImplementedError
