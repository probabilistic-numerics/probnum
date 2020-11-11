"""
Gaussian filtering and smoothing.
"""

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.gaussfiltsmooth.kalmanposterior import KalmanPosterior
from probnum.random_variables import Normal
from probnum.filtsmooth.gaussfiltsmooth.stoppingcriterion import StoppingCriterion


class Kalman(BayesFiltSmooth):
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

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the filtered output
        """
        dataset, times = np.asarray(dataset), np.asarray(times)
        filtrv = self.initialrandomvariable
        rvs = [filtrv]
        for idx in range(1, len(times)):
            filtrv, _ = self.filter_step(
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
        info = {}
        predrv, info["info_pred"] = self.predict(start, stop, randvar)
        filtrv, info["meas_rv"], info["info_upd"] = self.update(stop, predrv, data)
        return filtrv, info

    def predict(self, start, stop, randvar, **kwargs):
        return self.dynamod.transition_rv(randvar, start, stop=stop, **kwargs)

    def update(self, time, randvar, data, **kwargs):
        """

        Parameters
        ----------
        time
        randvar
        data
        kwargs

        Returns
        -------
        Normal
            Updated Normal RV (new filter estimate).
        Normal
            Measured random variable, as returned by the measurement model.
        dict
            Additional info. Contains at least the key `crosscov`,
            which is the crosscov between input RV and measured RV.
            The crosscov does not relate to the updated RV!
        """
        meas_rv, info = self.measmod.transition_rv(randvar, time, **kwargs)
        crosscov = info["crosscov"]
        new_mean = randvar.mean + crosscov @ np.linalg.solve(
            meas_rv.cov, data - meas_rv.mean
        )
        new_cov = randvar.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
        return Normal(new_mean, new_cov), meas_rv, info

    def smooth(self, filter_posterior, **kwargs):
        """
        Apply Gaussian smoothing to the filtering outcome (i.e. a KalmanPosterior).

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

    def smooth_list(self, rv_list, locations, **kwargs):
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

        Returns
        -------
        _RandomVariableList
            List of smoothed random variables.
        """
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
        smoothed_rv : RandomVariable
            Prediction at time t+1 of the filtering distribution at time t.
        start : float
            Time-point of the to-be-smoothed RV.
        stop : float
            Time-point of the already-smoothed RV.
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


class IteratedKalman(Kalman):
    """Iterated filter/smoother based on posterior linearisation.


    In principle, this is the same as a Kalman filter; however, there is also
    iterated_filtsmooth(), which computes things like MAP estimates.
    """

    def __init__(self, kalman, stoppingcriterion=None):
        self.kalman = kalman
        if stoppingcriterion is None:
            self.stoppingcriterion = StoppingCriterion()
        else:
            self.stoppingcriterion = stoppingcriterion
        super().__init__(kalman.dynamod, kalman.measmod, kalman.initrv)

    def filter_step(self, start, stop, current_rv, data, linearise_at=None, **kwargs):
        if linearise_at is None:
            filt_rv, info = self.kalman.filter_step(
                current_rv, data, start, stop, **kwargs
            )
            pred_rv = info["pred_rv"]
            meas_rv = info["meas_rv"]
            info_pred = info["info_pred"]
            info_upd = info["info_upd"]
        else:
            data = np.asarray(data)
            pred_rv, info_pred = self.predict(
                start, stop, current_rv, linearise_at=linearise_at
            )
            filt_rv, meas_rv, info_upd = self.update(
                stop, pred_rv, data, linearise_at=linearise_at
            )

        # repeat until happy
        if self.continue_filter_updates(pred_rv, info_pred, filt_rv, meas_rv, info_upd):
            return self.filter_step(start, stop, filt_rv, data, linearise_at=filt_rv)

    #
    # def iterated_filtsmooth(self):
    #     out = self.filtsmooth()
    #     while not self.stoppingcriterion.stop_filtsmooth_updates():
    #         out = self.filtsmooth(out)
