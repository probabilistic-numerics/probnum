"""
Gaussian filtering.
"""
from abc import ABC, abstractmethod

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.gaussfiltsmooth.kalmanposterior import KalmanPosterior
from probnum.random_variables import Normal


class Kalman(BayesFiltSmooth, ABC):
    """
    Gaussian filtering and smoothing, i.e. Kalman-like filters and smoothers.
    """

    def __init__(self, dynamic_model, measurement_model, initrv):
        """Check that the initial distribution is Gaussian."""
        if not issubclass(type(initrv), Normal):
            raise ValueError(
                "Gaussian filters/smoothers need initial "
                "random variables with Normal distribution."
            )
        super().__init__(dynamic_model, measurement_model, initrv)

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
        KalmanPosterior
            Posterior distribution of the smoothed output
        """
        dataset, times = np.asarray(dataset), np.asarray(times)
        filter_posterior = self.filter(dataset, times, **kwargs)
        smooth_posterior = self.smooth(filter_posterior, **kwargs)
        return smooth_posterior

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
        KalmanPosterior
            Posterior distribution of the filtered output
        """
        dataset, times = np.asarray(dataset), np.asarray(times)
        filtrv = self.initialrandomvariable
        rvs = [filtrv]
        for idx in range(1, len(times)):
            filtrv = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                randvar=filtrv,
                data=dataset[idx - 1],
            )
            rvs.append(filtrv)
        return KalmanPosterior(times, rvs, self, with_smoothing=False)

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
            Predict based on this random variable. For instance, this can be the result
            of a previous call to filter_step.
        data : array_like
            Compute the update based on this data.

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        """
        data = np.asarray(data)
        predrv, _ = self.predict(start, stop, randvar)
        filtrv, _, _, _ = self.update(stop, predrv, data)
        return filtrv

    def predict(self, start, stop, randvar, **kwargs):
        return self.dynamod.transition_rv(randvar, start, stop=stop, **kwargs)

    def update(self, time, randvar, data, **kwargs):
        meas_rv, info = self.measmod.transition_rv(randvar, time, **kwargs)
        crosscov = info["crosscov"]
        new_mean = randvar.mean + crosscov @ np.linalg.solve(
            meas_rv.cov, data - meas_rv.mean
        )
        new_cov = randvar.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
        return Normal(new_mean, new_cov), meas_rv.cov, crosscov, meas_rv.mean

    def smooth(self, filter_posterior, **kwargs):
        """
        Apply Gaussian smoothing to a set of filtered means and covariances.

        Parameters
        ----------
        filter_posterior : KalmanPosterior
            Posterior distribution obtained after filtering

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the smoothed output
        """
        rv_list = self.smooth_list(
            filter_posterior, filter_posterior.locations, **kwargs
        )
        return KalmanPosterior(
            filter_posterior.locations, rv_list, self, with_smoothing=True
        )

    def smooth_list(self, rv_list, locations, final_rv=None, **kwargs):
        """
        Apply smoothing to a list of RVs with desired final random variable.

        Specification of a final RV is useful to compute joint samples from a KalmanPosterior object,
        because in this case, the final RV is a Dirac (over a sample from the final Normal RV)
        and not a Normal RV.

        Parameters
        ----------
        rv_list : _RandomVariableList or array_like
            List of random variables to be smoothed.
        locations : array_like
            Locations of the random variables in rv_list.
        final_rv : RandomVariable, optional.
            RandomVariable at the final point. Default is None, in which case standard smoothing is applied.
            If a random variable is specified, the smoothing iteration is based on this one, which is used
            for sampling (in which case the final random variable is a Dirac that represents a sample)

        Returns
        -------
        _RandomVariableList
            List of smoothed random variables.
        """
        if final_rv is None:
            final_rv = rv_list[-1]
        curr_rv = final_rv
        out_rvs = [curr_rv]
        for idx in reversed(range(1, len(locations))):
            unsmoothed_rv = rv_list[idx - 1]
            curr_rv = self.smooth_step(
                unsmoothed_rv,
                curr_rv,
                start=locations[idx - 1],
                stop=locations[idx],
                **kwargs
            )
            out_rvs.append(curr_rv)
        out_rvs.reverse()
        return _RandomVariableList(out_rvs)

    def smooth_step(self, unsmoothed_rv, smoothed_rv, start, stop, **kwargs):
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
        predicted_rv, info = self.dynamod.transition_rv(
            unsmoothed_rv, start, stop=stop, **kwargs
        )
        crosscov = info["crosscov"]
        smoothing_gain = np.linalg.solve(predicted_rv.cov.T, crosscov.T).T
        new_mean = unsmoothed_rv.mean + smoothing_gain @ (
            smoothed_rv.mean - predicted_rv.mean
        )
        new_cov = (
            unsmoothed_rv.cov
            + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
        )
        return Normal(new_mean, new_cov)
