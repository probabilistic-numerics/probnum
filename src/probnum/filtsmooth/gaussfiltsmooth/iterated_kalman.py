"""Iterated Gaussian filtering and smoothing."""
import numpy as np
from .kalman import Kalman
from .stoppingcriterion import StoppingCriterion
from .kalmanposterior import KalmanPosterior


class IteratedKalman(Kalman):
    """Iterated filter/smoother based on posterior linearisation.

    In principle, this is the same as a Kalman filter; however, there is
    also iterated_filtsmooth(), which computes things like MAP
    estimates.
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
        By further specifying a `previous_posterio` (KalmanPosterior), the first of those possibly
        iterated updates is linearised at the previous posterior estimate.
        This retrieves methods such as the iterated extended Kalman smoother.

        Parameters
        ----------
        start
        stop
        current_rv
        data
        previous_posterior :  KalmanPosterior

        Returns
        -------
        """
        info = {}
        # initial prediction
        if previous_posterior is None:
            pred_rv, info_pred = self.kalman.predict(start, stop, current_rv)
        else:
            pred_rv, info_pred = self.kalman.predict(
                start, stop, current_rv, linearise_at=previous_posterior(start)
            )

        # keep re-predicting while unhappy
        while self.stoppingcriterion.continue_predict_iteration(pred_rv, info_pred):
            pred_rv, info_pred = self.kalman.predict(
                start, stop, current_rv, linearise_at=pred_rv
            )

        info["pred_rv"] = pred_rv
        info["info_pred"] = info_pred

        # initial update
        if previous_posterior is None:
            upd_rv, meas_rv, info_upd = self.kalman.update(stop, current_rv, data)
        else:
            upd_rv, meas_rv, info_upd = self.kalman.update(
                stop, current_rv, data, linearise_at=previous_posterior(stop)
            )

        # keep re-updating while unhappy
        while self.stoppingcriterion.continue_update_iteration(
            upd_rv, meas_rv, info_upd
        ):
            upd_rv, meas_rv, info_upd = self.kalman.update(
                stop, current_rv, data, linearise_at=upd_rv
            )

        info["meas_rv"] = meas_rv
        info["info_upd"] = info_upd

        return upd_rv, info

    def iterated_filtsmooth(self, dataset, times, **kwargs):
        """Repeated filtering and smoothing using posterior linearisation."""
        posterior = self.filtsmooth(dataset, times, **kwargs)
        while not self.stoppingcriterion.continue_filtsmooth_updates(posterior):
            posterior = self.filter(dataset, times, linearise_at=posterior)
            posterior = self.smooth(posterior)
        return posterior
