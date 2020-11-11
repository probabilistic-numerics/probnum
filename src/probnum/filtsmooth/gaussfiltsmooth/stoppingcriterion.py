"""Stopping criteria for iterated filtering and smoothing."""
import numpy as np


class StoppingCriterion:
    """Stopping criteria for iterated filters/smoothers."""

    def __init__(
        self,
        max_num_predicts_per_step=None,
        max_num_updates_per_step=None,
        max_num_filtsmooth_iterations=None,
    ):
        self.num_predict_iterations = 0
        self.num_update_iterations = 0
        self.num_filtsmooth_iterations = 0

        self.max_num_predicts_per_step = max_num_predicts_per_step
        self.max_num_updates_per_step = max_num_updates_per_step
        self.max_num_filtsmooth_iterations = max_num_filtsmooth_iterations

    def continue_predict_iteration(self, pred_rv=None, info_pred=None):
        """Do we continue iterating the update step of the filter?"""
        return False

    def continue_update_iteration(self, upd_rv=None, meas_rv=None, info_upd=None):
        """Do we continue iterating the predict step of the filter?"""
        return False

    def continue_filtsmooth_iteration(self, kalman_posterior=None):
        """If implemented, iterated_filtsmooth() is unlocked."""
        raise NotImplementedError


class FixedPointStopping(StoppingCriterion):
    """Keep updating until the filter recursion arrives at a fixed-point."""

    def __init__(
        self,
        atol=1e-2,
        rtol=1e-2,
        max_num_predicts_per_step=1000,
        max_num_updates_per_step=1000,
        max_num_filtsmooth_iterations=1000,
    ):
        self.atol = atol
        self.rtol = rtol
        self.previous_pred_rv = None
        self.previous_upd_rv = None
        self.previous_posterior = None
        super().__init__(
            max_num_predicts_per_step=max_num_predicts_per_step,
            max_num_updates_per_step=max_num_updates_per_step,
            max_num_filtsmooth_iterations=max_num_filtsmooth_iterations,
        )

    def continue_predict_iteration(self, pred_rv=None, info_pred=None):
        """Do we continue iterating the update step of the filter?"""
        # Edge cases: zeroth or last iteration
        if self.num_predict_iterations >= self.max_num_predicts_per_step:
            raise RuntimeError("Maximum number of filter update iterations reached.")
        self.num_predict_iterations += 1
        if self.previous_pred_rv is None:
            self.previous_pred_rv = pred_rv
            return True

        # Compute relative thresholds
        mean_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_pred_rv.mean), np.abs(pred_rv.mean)
        )
        cov_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_pred_rv.cov), np.abs(pred_rv.cov)
        )

        # Accept if discrepancy sufficiently small
        mean_acceptable = np.all(
            np.abs(pred_rv.mean - self.previous_pred_rv.mean) < mean_threshold
        )
        cov_acceptable = np.all(
            np.abs(pred_rv.cov - self.previous_pred_rv.cov) < cov_threshold
        )
        continue_iteration = np.invert(
            np.all(np.logical_and(mean_acceptable, cov_acceptable))
        )
        if continue_iteration:
            self.previous_pred_rv = pred_rv
        return continue_iteration

    def continue_update_iteration(self, upd_rv=None, meas_rv=None, info_upd=None):
        """Do we continue iterating the predict step of the filter?"""
        # Edge cases: zeroth or last iteration
        if self.num_update_iterations >= self.max_num_updates_per_step:
            raise RuntimeError("Maximum number of filter update iterations reached.")
        self.num_update_iterations += 1
        if self.previous_upd_rv is None:
            self.previous_upd_rv = upd_rv
            return True

        # Compute relative thresholds
        mean_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_upd_rv.mean), np.abs(upd_rv.mean)
        )
        cov_threshold = self.atol + self.rtol * np.maximum(
            np.abs(self.previous_upd_rv.cov), np.abs(upd_rv.cov)
        )

        # Accept if discrepancy sufficiently small
        mean_acceptable = np.all(
            np.abs(upd_rv.mean - self.previous_upd_rv.mean) < mean_threshold
        )
        cov_acceptable = np.all(
            np.abs(upd_rv.cov - self.previous_upd_rv.cov) < cov_threshold
        )
        continue_iteration = np.invert(
            np.all(np.logical_and(mean_acceptable, cov_acceptable))
        )
        if continue_iteration:
            self.previous_upd_rv = upd_rv
        return continue_iteration

    def continue_filtsmooth_iteration(self, kalman_posterior=None):

        # Edge cases: zeroth or last iteration
        if self.num_filtsmooth_iterations >= self.max_num_filtsmooth_iterations:
            raise RuntimeError("Maximum number of filter update iterations reached.")
        self.num_filtsmooth_iterations += 1
        if self.previous_posterior is None:
            self.previous_posterior = kalman_posterior
            return True

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
