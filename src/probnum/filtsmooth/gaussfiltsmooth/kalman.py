"""Gaussian filtering and smoothing."""

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.gaussfiltsmooth.kalmanposterior import KalmanPosterior
from probnum.random_variables import Normal


class Kalman(BayesFiltSmooth):
    """Gaussian filtering and smoothing, i.e. Kalman-like filters and smoothers."""

    def __init__(self, dynamic_model, measurement_model, initrv):
        """Check that the initial distribution is Gaussian."""
        if not issubclass(type(initrv), Normal):
            raise ValueError(
                "Gaussian filters/smoothers need initial "
                "random variables with Normal distribution."
            )
        super().__init__(dynamic_model, measurement_model, initrv)

    def filtsmooth(self, dataset, times, **kwargs):
        """Apply Gaussian filtering and smoothing to a data set.

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
        """Apply Gaussian filtering (no smoothing!) to a data set.

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
                current_rv=filtrv,
                data=dataset[idx - 1],
                **kwargs
            )
            rvs.append(filtrv)
        return KalmanPosterior(times, rvs, self, with_smoothing=False)

    def filter_step(self, start, stop, current_rv, data, **kwargs):
        """A single filter step.

        Consists of a prediction step (t -> t+1) and an update step (at t+1).

        Parameters
        ----------
        start : float
            Predict FROM this time point.
        stop : float
            Predict TO this time point.
        current_rv : RandomVariable
            Predict based on this random variable. For instance, this can be the result
            of a previous call to filter_step.
        data : array_like
            Compute the update based on this data.

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        dict
            Additional information provided by predict() and update().
            Contains keys `pred_rv`, `info_pred`, `meas_rv`, `info_upd`.
        """
        data = np.asarray(data)
        info = {}
        info["pred_rv"], info["info_pred"] = self.predict(
            start, stop, current_rv, **kwargs
        )
        filtrv, info["meas_rv"], info["info_upd"] = self.update(
            stop, info["pred_rv"], data, **kwargs
        )
        return filtrv, info

    def predict(self, start, stop, randvar, **kwargs):
        return self.dynamod.transition_rv(randvar, start, stop=stop, **kwargs)

    def measure(self, time, randvar, **kwargs):
        """Propagate the state through the measurement model.

        Parameters
        ----------
        time : float
            Time of the measurement.
        randvar : Normal
            Random variable to be propagated through the measurement model.

        Returns
        -------
        meas_rv : Normal
            Measured random variable, as returned by the measurement model.
        info : dict
            Additional info. Contains at leas the key `crosscov` which is the cross
            covariance between the input random variable and the measured random
            variable.
        """
        return self.measmod.transition_rv(randvar, time, **kwargs)

    def condition_state_on_measurement(
        self, randvar, meas_rv, data, crosscov, **kwargs
    ):
        """Condition the state on the observed data.

        Parameters
        ----------
        randvar : Normal
            Random variable to be updated with the measurement and data.
        meas_rv : Normal
            Measured random variable, as returned by the measurement model.
        data : np.ndarray
            Data to update on.
        crosscov : np.ndarray
            Cross-covariance between the state random variable `randvar` and the
            measurement random variable `meas_rv`.

        Returns
        -------
        Normal
            Updated Normal random variable (new filter estimate)
        """
        new_mean = randvar.mean + crosscov @ np.linalg.solve(
            meas_rv.cov, data - meas_rv.mean
        )
        new_cov = randvar.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
        filt_rv = Normal(new_mean, new_cov)
        return filt_rv

    def update(self, time, randvar, data, **kwargs):
        """Gaussian filter update step. Consists of a measurement step and a
        conditioning step.

        Parameters
        ----------
        time : float
            Time of the update.
        randvar : RandomVariable
            Random variable to be updated. Result of :meth:`predict()`.
        data : np.ndarray
            Data to update on.

        Returns
        -------
        filt_rv : Normal
            Updated Normal RV (new filter estimate).
        meas_rv : Normal
            Measured random variable, as returned by the measurement model.
        info : dict
            Additional info. Contains at least the key `crosscov`,
            which is the crosscov between input RV and measured RV.
            The crosscov does not relate to the updated RV!
        """
        meas_rv, info = self.measure(time, randvar, **kwargs)
        filt_rv = self.condition_state_on_measurement(
            randvar, meas_rv, data, info["crosscov"], **kwargs
        )
        return filt_rv, meas_rv, info

    def smooth(self, filter_posterior, **kwargs):
        """Apply Gaussian smoothing to the filtering outcome (i.e. a KalmanPosterior).

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
        """Apply smoothing to a list of RVs with desired final random variable.

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
        """A single smoother step.

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
