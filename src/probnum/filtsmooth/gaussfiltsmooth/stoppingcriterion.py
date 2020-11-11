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

    def __init__(self, max_num_filter_updates=None, max_num_filtsmooth_updates=None):
        self.num_filter_updates = 0
        self.num_filtsmooth_updates = 0
        self.max_num_filter_updates = max_num_filter_updates
        self.max_num_filtsmooth_updates = max_num_filtsmooth_updates

    def continue_filter_updates(
        self, predrv=None, info_pred=None, filtrv=None, meas_rv=None, info_upd=None
    ):
        """
        When do we stop iterating the filter steps. Default is true.
        If, e.g. IEKF is wanted, overwrite with something that does not always return True.
        """
        return False

    def continue_filtsmooth_updates(self, kalman_posterior=None):
        """If implemented, iterated_filtsmooth() is unlocked."""
        raise NotImplementedError


class FixedPointStopping(StoppingCriterion):
    """Keep updating until the filter recursion arrives at a fixed-point."""

    def __init__(
        self,
        atol=1e-2,
        rtol=1e-2,
        max_num_filter_updates=1000,
        max_num_filtsmooth_updates=1000,
    ):
        self.atol = atol
        self.rtol = rtol
        self.previous_rv = None
        self.previous_posterior = None
        super().__init__(
            max_num_filter_updates=max_num_filter_updates,
            max_num_filtsmooth_updates=max_num_filtsmooth_updates,
        )

    def continue_filter_updates(
        self, pred_rv=None, info_pred=None, filt_rv=None, meas_rv=None, info_upd=None
    ):
        """Continue filtering updates unless a fixed-point was found."""

        # Edge cases: zeroth or last iteration
        if self.num_filter_updates >= self.max_num_filter_updates:
            raise RuntimeError("Maximum number of filter update iterations reached.")
        self.num_filter_updates += 1
        if self.previous_rv is None:
            self.previous_rv = filt_rv
            return True

        # Compute relative thresholds
        mean_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_rv.mean), np.abs(filt_rv.mean)
        )
        cov_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_rv.cov), np.abs(filt_rv.cov)
        )

        # Accept if discrepancy sufficiently small
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

    def continue_filtsmooth_updates(self, kalman_posterior=None):

        # Edge cases: zeroth or last iteration
        if self.num_filtsmooth_updates >= self.max_num_filtsmooth_updates:
            raise RuntimeError("Maximum number of filter update iterations reached.")
        self.num_filtsmooth_updates += 1
        if self.previous_rv is None:
            self.previous_posterior = kalman_posterior
            return True

        #######################################################################################
        # below is experimental...
        # I think it works, but it is not tested...
        #######################################################################################

        # Compute relative thresholds
        mean_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_posterior.state_rvs.mean),
            np.abs(kalman_posterior.state_rvs.mean),
        )
        cov_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_posterior.state_rvs.cov),
            np.abs(kalman_posterior.state_rvs.cov),
        )

        # Accept if discrepancy sufficiently small
        mean_acceptable = np.all(
            np.abs(
                kalman_posterior.state_rvs.mean - self.previous_posterior.state_rvs.mean
            )
            < mean_threshold
        )
        cov_acceptable = np.all(
            np.abs(
                kalman_posterior.state_rvs.cov - self.previous_posterior.state_rvs.cov
            )
            < cov_threshold
        )
        continue_iteration = np.invert(
            np.all(np.logical_and(mean_acceptable, cov_acceptable))
        )
        if continue_iteration:
            self.previous_posterior = kalman_posterior
        return continue_iteration
