"""Iterated Gaussian filtering and smoothing."""
import numpy as np

import probnum.random_variables as pnrv

from .kalman import Kalman
from .stoppingcriterion import StoppingCriterion


class IteratedKalman(Kalman):
    """Iterated filter/smoother based on posterior linearisation.

    In principle, this is the same as a Kalman filter; however,
    1. predict() and update() in each filter step may be repeated.
    2. approximate Gaussian filtering and smoothing can use posterior linearisation.

    There is an additional method: :meth:`iterated_filtsmooth()`,
    which computes things like MAP estimates.
    """

    def __init__(self, kalman, stoppingcriterion=None):
        self.kalman = kalman
        if stoppingcriterion is None:
            self.stoppingcriterion = StoppingCriterion()
        else:
            self.stoppingcriterion = stoppingcriterion
        super().__init__(kalman.dynamod, kalman.measmod, kalman.initrv)

    def filter_step(
        self, start, stop, current_rv, data, previous_posterior=None, **kwargs
    ):
        """Filter step of iterated filter.

        Different to Kalman.filter in the sense that there may be multiple iterated updates for
        a single predict, or multiple iterated predicts for a single update, or multiple iterated predicts
        and updates in general.
        This retrieves methods such as the iterated extended Kalman filter.
        By further specifying a `previous_posterior` (KalmanPosterior), the first of those possibly
        iterated updates is linearised at the previous posterior estimate.
        This retrieves methods such as the iterated extended Kalman smoother.

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
        previous_posterior : KalmanPosterior
            Posterior distribution of a previous smoothing iteration. Optional.
            If specified, posterior linearisation is applied.

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        dict
            Additional information provided by predict() and update().
            Contains keys `pred_rv`, `info_pred`, `meas_rv`, `info_upd`.
        """

        # initial prediction
        if previous_posterior is None:
            linearise_predict_at = None
            linearise_update_at = None
        else:
            linearise_predict_at = previous_posterior(start)
            linearise_update_at = previous_posterior(stop)
        pred_rv, info_pred = self.predict(
            start, stop, current_rv, linearise_at=linearise_predict_at
        )
        upd_rv, meas_rv, info_upd = self.update(
            stop, pred_rv, data, linearise_at=linearise_update_at
        )

        # Specify additional information to be returned
        info = {
            "pred_rv": pred_rv,
            "info_pred": info_pred,
            "meas_rv": meas_rv,
            "info_upd": info_upd,
        }
        return upd_rv, info

    def iterated_filtsmooth(self, dataset, times, **kwargs):
        """Repeated filtering and smoothing using posterior linearisation."""
        posterior = self.filtsmooth(dataset, times, **kwargs)
        while self.stoppingcriterion.continue_filtsmooth_iteration(posterior):
            posterior = self.filter(dataset, times, linearise_at=posterior)
            posterior = self.smooth(posterior)
        return posterior

    def predict(self, start, stop, randvar, linearise_at=None, **kwargs):
        """(Possibly iterated) prediction step."""
        pred_rv, info_pred = self.dynamod.transition_rv(
            randvar, start, stop=stop, linearise_at=linearise_at, **kwargs
        )
        while self.stoppingcriterion.continue_predict_iteration(pred_rv, info_pred):
            pred_rv, info_pred = self.dynamod.transition_rv(
                pred_rv, start, stop, linearise_at=pred_rv
            )
        return pred_rv, info_pred

    def update(self, time, randvar, data, linearise_at=None, **kwargs):
        """(Possibly iterated) update step."""
        upd_rv, meas_rv, info_upd = self._single_update(
            time, randvar, data, linearise_at=linearise_at, **kwargs
        )
        while self.stoppingcriterion.continue_update_iteration(
            upd_rv, meas_rv, info_upd
        ):
            upd_rv, meas_rv, info_upd = self._single_update(
                time, randvar, data, linearise_at=upd_rv, **kwargs
            )
        return upd_rv, meas_rv, info_upd

    def _single_update(self, time, randvar, data, linearise_at=None, **kwargs):
        # like kalman.update but with an explicit linearise_at argument
        meas_rv, info = self.measmod.transition_rv(
            randvar, time, linearise_at=linearise_at, **kwargs
        )
        crosscov = info["crosscov"]
        new_mean = randvar.mean + crosscov @ np.linalg.solve(
            meas_rv.cov, data - meas_rv.mean
        )
        new_cov = randvar.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
        filt_rv = pnrv.Normal(new_mean, new_cov)
        return filt_rv, meas_rv, info
