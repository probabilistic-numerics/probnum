"""Square-root Gaussian filtering and smoothing."""

import typing

import numpy as np
import scipy.linalg

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv
import probnum.type as pntype

from .extendedkalman import DiscreteEKFComponent
from .kalman import Kalman
from .sqrt_utils import cholesky_update, sqrt_kalman_update, sqrt_smoothing_step


# This class re-implements predict and measure and does not use Transition.transition_rv,
# because e.g. update and smoothing_step MUST be in here.
# Therefore it seems better to do everything here.
class SquareRootKalman(Kalman):
    def __init__(
        self,
        dynamics_model: typing.Union[
            pnfss.LTISDE, pnfss.DiscreteLinearGaussian, DiscreteEKFComponent
        ],
        measurement_model: typing.Union[
            pnfss.DiscreteLinearGaussian, DiscreteEKFComponent
        ],
        initrv: pnrv.Normal,
    ) -> None:
        """Check that the models are linear(ised)."""
        # EKF is acceptable, because of the Taylor linearisation.
        # UKF would need downdates, which is not supported (at the moment?).

        compatible_dynamics_models = (
            pnfss.LTISDE,
            pnfss.DiscreteLinearGaussian,
            DiscreteEKFComponent,
        )
        if not isinstance(
            dynamics_model,
            compatible_dynamics_models,
        ):
            errormsg = (
                "The dynamics model must be able to reduce to a linear discrete model."
            )
            raise ValueError(errormsg)

        compatible_measurement_models = (
            pnfss.DiscreteLinearGaussian,
            DiscreteEKFComponent,
        )
        if not isinstance(measurement_model, compatible_measurement_models):
            errormsg = "The measurement model must be able to reduce to a linear discrete model."
            raise ValueError(errormsg)

        super().__init__(dynamics_model, measurement_model, initrv)

    def predict(
        self,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        randvar: pnrv.Normal,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        dynamicsmat, forcevec, diffmat_cholesky = self._linear_dynamic_matrices(
            start, stop, randvar
        )

        new_mean = dynamicsmat @ randvar.mean + forcevec
        new_cov_cholesky = cholesky_update(
            dynamicsmat @ randvar.cov_cholesky, diffmat_cholesky
        )
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        crosscov = randvar.cov @ dynamicsmat.T

        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), {
            "crosscov": crosscov
        }

    def measure(
        self, time: pntype.FloatArgType, randvar: pnrv.Normal
    ) -> (pnrv.Normal, typing.Dict):
        dynamicsmat, forcevec, diffmat_cholesky = self._linear_measurement_matrices(
            time, randvar
        )

        new_mean = dynamicsmat @ randvar.mean + forcevec
        new_cov_cholesky = cholesky_update(
            dynamicsmat @ randvar.cov_cholesky, diffmat_cholesky
        )
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        crosscov = randvar.cov @ dynamicsmat.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), {
            "crosscov": crosscov
        }

    def update(
        self, time: pntype.FloatArgType, randvar: pnrv.Normal, data: np.ndarray
    ) -> (pnrv.Normal, pnrv.Normal, typing.Dict):

        dynamicsmat, forcevec, diffmat_cholesky = self._linear_measurement_matrices(
            time, randvar
        )

        meascov_cholesky, kalman_gain, new_cov_cholesky = sqrt_kalman_update(
            dynamicsmat, diffmat_cholesky, randvar.cov_cholesky
        )

        measrv_mean = dynamicsmat @ randvar.mean + forcevec
        res = data - measrv_mean
        new_mean = randvar.mean + kalman_gain @ res
        new_cov = new_cov_cholesky @ new_cov_cholesky.T

        meas_rv = pnrv.Normal(
            measrv_mean,
            cov=meascov_cholesky @ meascov_cholesky.T,
            cov_cholesky=meascov_cholesky,
        )

        return (
            pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky),
            meas_rv,
            {},
        )

    def _linear_measurement_matrices(
        self, time: pntype.FloatArgType, randvar: pnrv.Normal
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        if isinstance(self.measurement_model, DiscreteEKFComponent):
            self.measurement_model.linearize(at_this_rv=randvar)
            disc_model = self.measurement_model.linearized_model
        else:
            disc_model = self.measurement_model

        dynamicsmat = disc_model.dynamicsmatfun(time)
        forcevec = disc_model.forcevecfun(time)
        diffmat_cholesky = disc_model.diffmatfun_cholesky(time)
        return dynamicsmat, forcevec, diffmat_cholesky

    def smooth_step(
        self,
        unsmoothed_rv: pnrv.Normal,
        smoothed_rv: pnrv.Normal,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        intermediate_step: typing.Optional[pntype.FloatArgType] = None,
    ):

        # I do not like that this prediction step is necessary.
        # Very soon we should start caching predictions.
        pred_rv, info = self.predict(start, stop, unsmoothed_rv)
        crosscov = info["crosscov"]
        smoothing_gain = scipy.linalg.cho_solve(
            (pred_rv.cov_cholesky, True), crosscov.T
        ).T

        dynamicsmat, forcevec, diffmat_cholesky = self._linear_dynamic_matrices(
            start, stop, unsmoothed_rv
        )
        new_cov_cholesky = sqrt_smoothing_step(
            unsmoothed_rv.cov_cholesky,
            dynamicsmat,
            diffmat_cholesky,
            smoothed_rv.cov_cholesky,
            smoothing_gain,
        )

        new_mean = unsmoothed_rv.mean + smoothing_gain @ (
            smoothed_rv.mean - pred_rv.mean
        )
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky)

    def _linear_dynamic_matrices(
        self,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        randvar: pnrv.Normal,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        if isinstance(self.dynamics_model, pnfss.LTISDE):
            disc_model = self.dynamics_model.discretise(stop - start)
            dynamicsmat = disc_model.driftmat
            forcevec = disc_model.forcevec
            diffmat_cholesky = disc_model.diffmat_cholesky
        elif isinstance(self.dynamics_model, DiscreteEKFComponent):
            self.dynamics_model.linearize(at_this_rv=randvar)
            disc_model = self.dynamics_model.linearized_model
            dynamicsmat = disc_model.dynamicsmatfun(start)
            forcevec = disc_model.forcevecfun(start)
            diffmat_cholesky = disc_model.diffmatfun_cholesky(start)
        else:  # must be discrete linear Gaussian model
            dynamicsmat = self.dynamics_model.dynamicsmatfun(start)
            forcevec = self.dynamics_model.forcevecfun(start)
            diffmat_cholesky = self.dynamics_model.diffmatfun_cholesky(start)
        return dynamicsmat, forcevec, diffmat_cholesky
