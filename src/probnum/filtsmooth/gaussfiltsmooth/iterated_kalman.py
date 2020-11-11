"""Iterated Gaussian filtering and smoothing."""

from .kalman import Kalman
from .stoppingcriterion import StoppingCriterion


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
        """Linearise at: KalmanPosterior object."""
        if linearise_at is None:
            filt_rv, info = self.kalman.filter_step(
                start, stop, current_rv, data, **kwargs
            )
            pred_rv = info["pred_rv"]
            meas_rv = info["meas_rv"]
            info_pred = info["info_pred"]
            info_upd = info["info_upd"]
        else:
            data = np.asarray(data)
            pred_rv, info_pred = self.predict(
                start, stop, current_rv, linearise_at=linearise_at(start)
            )
            filt_rv, meas_rv, info_upd = self.update(
                stop,
                pred_rv,
                data,
                linearise_at=linearise_at(
                    stop
                ),  # should this be a posterior object or a Normal??
            )

        # repeat until happy
        if self.stoppingcriterion.continue_filter_updates(
            pred_rv, info_pred, filt_rv, meas_rv, info_upd
        ):
            return self.filter_step(start, stop, filt_rv, data, linearise_at=filt_rv)

    def iterated_filtsmooth(self, dataset, times, **kwargs):
        """Repeated filtering and smoothing using posterior linearisation."""
        posterior = self.filtsmooth(dataset, times, **kwargs)
        while not self.stoppingcriterion.continue_filtsmooth_updates(posterior):
            posterior = self.filter(dataset, times, linearise_at=posterior)
            posterior = self.smooth(posterior)
        return posterior
