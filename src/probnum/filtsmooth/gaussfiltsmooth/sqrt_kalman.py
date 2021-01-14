"""Square-root Gaussian filtering and smoothing."""

import numpy as np
import scipy.linalg

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv

from .extendedkalman import DiscreteEKFComponent
from .kalman import Kalman
from .sqrt_utils import cholesky_update, sqrt_kalman_update, sqrt_smoothing_step


# This class re-implements predict and measure and does not use Transition.transition_rv,
# because e.g. update and smoothing_step MUST be in here.
# Therefore it seems better to do everything here.
class SquareRootKalman(Kalman):
    def __init__(self, dynamics_model, measurement_model, initrv):
        """Check that the models are linear(ised)."""
        # EKF is acceptable, because of the Taylor linearisation.
        # UKF would need downdates, which is not supported (at the moment?).

        if not isinstance(
            dynamics_model,
            (pnfss.LTISDE, pnfss.DiscreteLinearGaussian, DiscreteEKFComponent),
        ):
            errormsg = (
                "The dynamics model must be able to reduce to a linear discrete model."
            )
            raise ValueError(errormsg)

        if not isinstance(
            measurement_model, (pnfss.DiscreteLinearGaussian, DiscreteEKFComponent)
        ):
            errormsg = "The measurement model must be able to reduce to a linear discrete model."
            raise ValueError(errormsg)

        super().__init__(dynamics_model, measurement_model, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        A, s, L_Q = self._linear_dynamic_matrices(start, stop, randvar)

        old_mean = randvar.mean
        old_cov_cholesky = randvar.cov_cholesky

        new_mean = A @ old_mean + s
        new_cov_cholesky = cholesky_update(A @ old_cov_cholesky, L_Q)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        crosscov = randvar.cov @ A.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), {
            "crosscov": crosscov
        }

    def _linear_dynamic_matrices(self, start, stop, randvar):
        if isinstance(self.dynamics_model, pnfss.LTISDE):
            disc_model = self.dynamics_model.discretise(stop - start)
            A = disc_model.driftmat
            s = disc_model.forcevec
            L_Q = disc_model.diffmat_cholesky
        elif isinstance(self.dynamics_model, DiscreteEKFComponent):
            self.dynamics_model.linearize(at_this_rv=randvar)
            disc_model = self.dynamics_model.linearized_model
            A = disc_model.dynamicsmatfun(start)
            s = disc_model.forcevecfun(start)
            L_Q = disc_model.diffmatfun_cholesky(start)
        else:  # must be discrete linear Gaussian model
            A = self.dynamics_model.dynamicsmatfun(start)
            s = self.dynamics_model.forcevecfun(start)
            L_Q = self.dynamics_model.diffmatfun_cholesky(start)
        return A, s, L_Q

    def _linear_measurement_matrices(self, time, randvar):
        if isinstance(self.measurement_model, DiscreteEKFComponent):
            self.measurement_model.linearize(at_this_rv=randvar)
            disc_model = self.measurement_model.linearized_model
        else:
            disc_model = self.measurement_model

        H = disc_model.dynamicsmatfun(time)
        s = disc_model.forcevecfun(time)
        L_R = disc_model.diffmatfun_cholesky(time)
        return H, s, L_R

    def measure(self, time, randvar):
        H, s, L_R = self._linear_measurement_matrices(time, randvar)
        old_mean = randvar.mean
        old_cov_cholesky = randvar.cov_cholesky

        new_mean = H @ old_mean + s
        new_cov_cholesky = cholesky_update(H @ old_cov_cholesky, L_R)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        crosscov = randvar.cov @ H.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), {
            "crosscov": crosscov
        }

    def update(self, time, randvar, data):
        predcov_cholesky = randvar.cov_cholesky
        H, s, L_R = self._linear_measurement_matrices(time, randvar)

        L_S, K, L_P = sqrt_kalman_update(H, L_R, predcov_cholesky)

        measrv_mean = H @ randvar.mean + s
        res = data - measrv_mean
        new_mean = randvar.mean + K @ res
        new_cov = L_P @ L_P.T

        meas_rv = pnrv.Normal(measrv_mean, cov=L_S @ L_S.T, cov_cholesky=L_S)

        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=L_P), meas_rv, {}

    def smooth_step(
        self, unsmoothed_rv, smoothed_rv, start, stop, intermediate_step=None
    ):

        # this is still very messy...

        A, s, L_Q = self._linear_dynamic_matrices(start, stop, unsmoothed_rv)

        pred_rv, info = self.predict(start, stop, unsmoothed_rv)
        crosscov = info["crosscov"]
        smoothing_gain = scipy.linalg.cho_solve(
            (pred_rv.cov_cholesky, True), crosscov.T
        ).T

        L_P = sqrt_smoothing_step(
            unsmoothed_rv.cov_cholesky, A, L_Q, smoothed_rv.cov_cholesky, smoothing_gain
        )

        new_mean = unsmoothed_rv.mean + smoothing_gain @ (
            smoothed_rv.mean - pred_rv.mean
        )
        new_cov = L_P @ L_P.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=L_P)
