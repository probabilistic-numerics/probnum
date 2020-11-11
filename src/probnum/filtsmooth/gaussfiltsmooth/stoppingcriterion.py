"""Stopping criteria for iterated filtering and smoothing."""
import numpy as np


class StoppingCriterion:
    """
    Stopping criteria for iterated filters/smoothers.


    By default this stopping criterion is defined in a way
    that iterated filters behave like normal filters.
    Though, to unlock `Kalman.iterated_filtsmooth`, implement `self.stop_filtsmooth_updates`.
    For iterated filtering (not iterated smoothing!), implement `self.stop_filter_updates`.
    """

    def continue_filter_updates(
        self,
        predrv=None,
        info_pred=None,
        filtrv=None,
        meas_rv=None,
        info_upd=None,
        **kwargs
    ):
        """
        When do we stop iterating the filter steps. Default is true.
        If, e.g. IEKF is wanted, overwrite with something that does not always return True.
        """
        return False

    def continue_filtsmooth_updates(self, **kwargs):
        """If implemented, iterated_filtsmooth() is unlocked."""
        raise NotImplementedError


class FixedPointStopping(StoppingCriterion):
    """Keep updating until the filter recursion arrives at a fixed-point."""

    def __init__(self, atol=1e-2, rtol=1e-2):
        self.atol = atol
        self.rtol = rtol
        self.previous_rv = None

    def continue_filter_updates(
        self,
        pred_rv=None,
        info_pred=None,
        filt_rv=None,
        meas_rv=None,
        info_upd=None,
        **kwargs
    ):
        """Continue filtering updates unless a fixed-point was found."""
        mean_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_rv.mean), np.abs(filt_rv.mean)
        )
        cov_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_rv.cov), np.abs(filt_rv.cov)
        )

        mean_acceptable = np.all(
            np.abs(filt_rv.mean - self.previous_rv.mean) < mean_threshold
        )
        cov_acceptable = np.all(
            np.abs(filt_rv.cov - self.previous_rv.cov) < cov_threshold
        )
        continue_iteration = np.invert(
            np.all(np.logical_and(mean_acceptable, cov_acceptable))
        )
        if continue_iteration:
            self.previous_rv = filt_rv
        return continue_iteration
