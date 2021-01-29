"""Gaussian filtering and smoothing."""

import functools as ft

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.gaussfiltsmooth.kalmanposterior import KalmanPosterior
from probnum.random_variables import Normal

from .kalman_utils import (
    measure_via_transition,
    predict_via_transition,
    rts_smooth_step_classic,
    update_classic,
)

#
#
# def iterated_filtsmooth(
#     kalman, stoppingcriterion, dataset, times, _intermediate_step=None
# ):
#     raise RuntimeError("TBD.")


class Kalman(BayesFiltSmooth):
    def __init__(
        self,
        dynamics_model,
        measurement_model,
        initrv,
        predict=predict_via_transition,
        measure=measure_via_transition,
        update=update_classic,
        smooth_step=rts_smooth_step_classic,
    ):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv
        self.predict = ft.partial(predict, dynamics_model=dynamics_model)
        self.measure = ft.partial(measure, measurement_model=measurement_model)
        self.update = ft.partial(update, measurement_model=measurement_model)
        self.smooth_step = smooth_step

    def filtsmooth(self, dataset, times, _intermediate_step=None, _linearise_at=None):
        dataset, times = np.asarray(dataset), np.asarray(times)
        filter_posterior = self.filter(
            dataset,
            times,
            _intermediate_step=_intermediate_step,
            _linearise_at=_linearise_at,
        )
        smooth_posterior = self.smooth(filter_posterior)
        return smooth_posterior

    def filter(self, dataset, times, _intermediate_step=None, _linearise_at=None):
        # _linearise_at is not used here, only in IteratedKalman.filter_step
        # which is overwritten by IteratedKalman
        dataset, times = np.asarray(dataset), np.asarray(times)
        rvs = []

        filtrv, *_ = self.update(rv=self.initrv, time=times[0], data=dataset[0])

        rvs.append(filtrv)
        for idx in range(1, len(times)):
            filtrv, _ = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                current_rv=filtrv,
                data=dataset[idx],
                _linearise_at=_linearise_at,
                _intermediate_step=_intermediate_step,
            )
            rvs.append(filtrv)
        return KalmanPosterior(times, rvs, self, with_smoothing=False)

    def filter_step(
        self,
        start,
        stop,
        current_rv,
        data,
        _intermediate_step=None,
        _linearise_at=None,
        _diffusion=1.0,
    ):
        data = np.asarray(data)
        info = {}
        info["pred_rv"], info["info_pred"] = self.predict(
            start=start,
            stop=stop,
            rv=current_rv,
            _intermediate_step=_intermediate_step,
            _diffusion=_diffusion,
        )
        filtrv, info["meas_rv"], info["info_upd"] = self.update(
            rv=info["pred_rv"], time=stop, data=data
        )
        return filtrv, info

    def smooth(self, filter_posterior):
        rv_list = self.smooth_list(
            filter_posterior,
            filter_posterior.locations,
        )
        return KalmanPosterior(
            filter_posterior.locations, rv_list, self, with_smoothing=True
        )

    def smooth_list(self, rv_list, locations):
        final_rv = rv_list[-1]
        curr_rv = final_rv
        out_rvs = [curr_rv]
        for idx in reversed(range(1, len(locations))):
            unsmoothed_rv = rv_list[idx - 1]

            # Intermediate prediction
            predicted_rv, info = self.predict(
                rv=unsmoothed_rv,
                start=locations[idx - 1],
                stop=locations[idx],
            )
            crosscov = info["crosscov"]
            smoothing_gain = crosscov @ np.linalg.inv(predicted_rv.cov)

            # Actual smoothing step
            curr_rv, _ = self.smooth_step(
                unsmoothed_rv, predicted_rv, curr_rv, smoothing_gain
            )
            out_rvs.append(curr_rv)
        out_rvs.reverse()
        return _RandomVariableList(out_rvs)


#
# class Kalman_(BayesFiltSmooth):
#     """Gaussian filtering and smoothing, i.e. Kalman-like filters and smoothers.
#
#     Parameters
#     ----------
#     dynamics_model
#         Prior dynamics. Usually an LTISDE object or an Integrator, but LinearSDE, ContinuousEKFComponent,
#         or ContinuousUKFComponent are also valid. Describes a random process in :math:`K` dimensions.
#         If an integrator, `K=spatialdim(ordint+1)` for some spatialdim and ordint.
#     measurement_model
#         Measurement model. Usually an DiscreteLTIGaussian, but any DiscreteLinearGaussian is acceptable.
#         This model maps the `K` dimensional prior state (see above) to the `L` dimensional space in which the observation ''live''.
#         For 2-dimensional observations, `L=2`.
#         If an DiscreteLTIGaussian, the measurement matrix is :math:`L \\times K` dimensional, the forcevec is `L` dimensional and the meascov is `L \\times L` dimensional.
#     initrv
#         Initial random variable for the prior. This is a `K` dimensional Gaussian distribution (not `L`, because it belongs to the prior)
#     """
#
#     def __init__(self, dynamics_model, measurement_model, initrv):
#         """Check that the initial distribution is Gaussian."""
#         if not issubclass(type(initrv), Normal):
#             raise ValueError(
#                 "Gaussian filters/smoothers need initial "
#                 "random variables with Normal distribution."
#             )
#         super().__init__(dynamics_model, measurement_model, initrv)
#
#     def filtsmooth(self, dataset, times, _intermediate_step=None):
#         """Apply Gaussian filtering and smoothing to a data set.
#
#         Parameters
#         ----------
#         dataset : array_like, shape (N, M)
#             Data set that is filtered.
#         times : array_like, shape (N,)
#             Temporal locations of the data points.
#             The zeroth element in times and dataset is the location of the initial random variable.
#
#         Returns
#         -------
#         KalmanPosterior
#             Posterior distribution of the smoothed output
#         """
#         dataset, times = np.asarray(dataset), np.asarray(times)
#         filter_posterior = self.filter(
#             dataset, times, _intermediate_step=_intermediate_step
#         )
#         smooth_posterior = self.smooth(
#             filter_posterior, _intermediate_step=_intermediate_step
#         )
#         return smooth_posterior
#
#     def filter(self, dataset, times, _intermediate_step=None, _linearise_at=None):
#         """Apply Gaussian filtering (no smoothing!) to a data set.
#
#         Parameters
#         ----------
#         dataset : array_like, shape (N, M)
#             Data set that is filtered.
#         times : array_like, shape (N,)
#             Temporal locations of the data points.
#             The zeroth element in times and dataset is the location of the initial random variable.
#
#         Returns
#         -------
#         KalmanPosterior
#             Posterior distribution of the filtered output
#         """
#         # _linearise_at is not used here, only in IteratedKalman.filter_step
#         # which is overwritten by IteratedKalman
#         dataset, times = np.asarray(dataset), np.asarray(times)
#         rvs = []
#
#         # initial update: since start=stop, prediction will not change the RV.
#         # This is relied on here but may lead to future problems.
#         filtrv, *_ = self.update(times[0], self.initrv, dataset[0])
#
#         rvs.append(filtrv)
#         for idx in range(1, len(times)):
#             filtrv, _ = self.filter_step(
#                 start=times[idx - 1],
#                 stop=times[idx],
#                 current_rv=filtrv,
#                 data=dataset[idx],
#                 _intermediate_step=_intermediate_step,
#             )
#             rvs.append(filtrv)
#         return KalmanPosterior(times, rvs, self, with_smoothing=False)
#
#     def filter_step(
#         self, start, stop, current_rv, data, _intermediate_step=None, _diffusion=1.0
#     ):
#         """A single filter step.
#
#         Consists of a prediction step (t -> t+1) and an update step (at t+1).
#
#         Parameters
#         ----------
#         start : float
#             Predict FROM this time point.
#         stop : float
#             Predict TO this time point.
#         current_rv : RandomVariable
#             Predict based on this random variable. For instance, this can be the result
#             of a previous call to filter_step.
#         data : array_like
#             Compute the update based on this data.
#
#         Returns
#         -------
#         RandomVariable
#             Resulting filter estimate after the single step.
#         dict
#             Additional information provided by predict() and update().
#             Contains keys `pred_rv`, `info_pred`, `meas_rv`, `info_upd`.
#         """
#         data = np.asarray(data)
#         info = {}
#         info["pred_rv"], info["info_pred"] = self.predict(
#             start,
#             stop,
#             current_rv,
#             _intermediate_step=_intermediate_step,
#             _diffusion=_diffusion,
#         )
#         filtrv, info["meas_rv"], info["info_upd"] = self.update(
#             stop, info["pred_rv"], data
#         )
#         return filtrv, info
#
#     def predict(self, start, stop, randvar, _intermediate_step=None, _diffusion=1.0):
#         return self.dynamics_model.transition_rv(
#             randvar,
#             start,
#             stop=stop,
#             step=_intermediate_step,
#             _diffusion=_diffusion,
#         )
#
#     def measure(self, time, randvar):
#         """Propagate the state through the measurement model.
#
#         Parameters
#         ----------
#         time : float
#             Time of the measurement.
#         randvar : Normal
#             Random variable to be propagated through the measurement model.
#
#         Returns
#         -------
#         meas_rv : Normal
#             Measured random variable, as returned by the measurement model.
#         info : dict
#             Additional info. Contains at leas the key `crosscov` which is the cross
#             covariance between the input random variable and the measured random
#             variable.
#         """
#         return self.measurement_model.transition_rv(randvar, time)
#
#     def condition_state_on_measurement(self, randvar, meas_rv, data, crosscov):
#         """Condition the state on the observed data.
#
#         Parameters
#         ----------
#         randvar : Normal
#             Random variable to be updated with the measurement and data.
#         meas_rv : Normal
#             Measured random variable, as returned by the measurement model.
#         data : np.ndarray
#             Data to update on.
#         crosscov : np.ndarray
#             Cross-covariance between the state random variable `randvar` and the
#             measurement random variable `meas_rv`.
#
#         Returns
#         -------
#         Normal
#             Updated Normal random variable (new filter estimate)
#         """
#         new_mean = randvar.mean + crosscov @ np.linalg.solve(
#             meas_rv.cov, data - meas_rv.mean
#         )
#         new_cov = randvar.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
#         filt_rv = Normal(new_mean, new_cov)
#         return filt_rv
#
#     def update(self, time, randvar, data):
#         """Gaussian filter update step. Consists of a measurement step and a
#         conditioning step.
#
#         Parameters
#         ----------
#         time : float
#             Time of the update.
#         randvar : RandomVariable
#             Random variable to be updated. Result of :meth:`predict()`.
#         data : np.ndarray
#             Data to update on.
#
#         Returns
#         -------
#         filt_rv : Normal
#             Updated Normal RV (new filter estimate).
#         meas_rv : Normal
#             Measured random variable, as returned by the measurement model.
#         info : dict
#             Additional info. Contains at least the key `crosscov`,
#             which is the crosscov between input RV and measured RV.
#             The crosscov does not relate to the updated RV!
#         """
#         meas_rv, info = self.measure(time, randvar)
#         filt_rv = self.condition_state_on_measurement(
#             randvar, meas_rv, data, info["crosscov"]
#         )
#         return filt_rv, meas_rv, info
#
#     def smooth(self, filter_posterior, _intermediate_step=None):
#         """Apply Gaussian smoothing to the filtering outcome (i.e. a KalmanPosterior).
#
#         Parameters
#         ----------
#         filter_posterior : KalmanPosterior
#             Posterior distribution obtained after filtering
#         _intermediate_step :
#             Step-size to be taken by approximate transition methods.
#         Returns
#         -------
#         KalmanPosterior
#             Posterior distribution of the smoothed output
#         """
#         rv_list = self.smooth_list(
#             filter_posterior,
#             filter_posterior.locations,
#             _intermediate_step=_intermediate_step,
#         )
#         return KalmanPosterior(
#             filter_posterior.locations, rv_list, self, with_smoothing=True
#         )
#
#     def smooth_list(self, rv_list, locations, _intermediate_step=None):
#         """Apply smoothing to a list of RVs with desired final random variable.
#
#         Specification of a final RV is useful to compute joint samples from a KalmanPosterior object,
#         because in this case, the final RV is a Dirac (over a sample from the final Normal RV)
#         and not a Normal RV.
#
#         Parameters
#         ----------
#         rv_list : _RandomVariableList or array_like
#             List of random variables to be smoothed.
#         locations : array_like
#             Locations of the random variables in rv_list.
#         _intermediate_step :
#             Step-size to be taken by approximate transition methods.
#
#         Returns
#         -------
#         _RandomVariableList
#             List of smoothed random variables.
#         """
#         final_rv = rv_list[-1]
#         curr_rv = final_rv
#         out_rvs = [curr_rv]
#         for idx in reversed(range(1, len(locations))):
#             unsmoothed_rv = rv_list[idx - 1]
#             curr_rv = self.smooth_step(
#                 unsmoothed_rv,
#                 curr_rv,
#                 start=locations[idx - 1],
#                 stop=locations[idx],
#                 _intermediate_step=_intermediate_step,
#             )
#             out_rvs.append(curr_rv)
#         out_rvs.reverse()
#         return _RandomVariableList(out_rvs)
#
#     def smooth_step(
#         self,
#         unsmoothed_rv,
#         smoothed_rv,
#         start,
#         stop,
#         _intermediate_step=None,
#         _diffusion=1.0,
#     ):
#         """A single smoother step.
#
#         Consists of predicting from the filtering distribution at time t
#         to time t+1 and then updating based on the discrepancy to the
#         smoothing solution at time t+1.
#         If preconditioning is available in the dynamic model, this is leveraged here.
#         If not, a classic smoothing step estimate is taken.
#
#         Parameters
#         ----------
#         unsmoothed_rv : RandomVariable
#             Filtering distribution at time t.
#         smoothed_rv : RandomVariable
#             Prediction at time t+1 of the filtering distribution at time t.
#         start : float
#             Time-point of the to-be-smoothed RV.
#         stop : float
#             Time-point of the already-smoothed RV.
#         _intermediate_step :
#             Step-size to be taken by approximate transition methods.
#         """
#         if self.dynamics_model.precon is None:
#             return self._smooth_step_classic(
#                 unsmoothed_rv,
#                 smoothed_rv,
#                 start,
#                 stop,
#                 _intermediate_step=_intermediate_step,
#                 _diffusion=_diffusion,
#             )
#         else:
#             return self._smooth_step_with_preconditioning(
#                 unsmoothed_rv,
#                 smoothed_rv,
#                 start,
#                 stop,
#                 _intermediate_step=_intermediate_step,
#                 _diffusion=_diffusion,
#             )
#
#     def _smooth_step_classic(
#         self,
#         unsmoothed_rv,
#         smoothed_rv,
#         start,
#         stop,
#         _intermediate_step=None,
#         _diffusion=1.0,
#     ):
#         """A single smoother step.
#
#         Consists of predicting from the filtering distribution at time t
#         to time t+1 and then updating based on the discrepancy to the
#         smoothing solution at time t+1.
#
#         Parameters
#         ----------
#         unsmoothed_rv : RandomVariable
#             Filtering distribution at time t.
#         smoothed_rv : RandomVariable
#             Prediction at time t+1 of the filtering distribution at time t.
#         start : float
#             Time-point of the to-be-smoothed RV.
#         stop : float
#             Time-point of the already-smoothed RV.
#         """
#         predicted_rv, info = self.dynamics_model.transition_rv(
#             unsmoothed_rv,
#             start,
#             stop=stop,
#             step=_intermediate_step,
#             _diffusion=_diffusion,
#         )
#         crosscov = info["crosscov"]
#         smoothing_gain = np.linalg.solve(predicted_rv.cov.T, crosscov.T).T
#         new_mean = unsmoothed_rv.mean + smoothing_gain @ (
#             smoothed_rv.mean - predicted_rv.mean
#         )
#         new_cov = (
#             unsmoothed_rv.cov
#             + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
#         )
#         return Normal(new_mean, new_cov)
#
#     def _smooth_step_with_preconditioning(
#         self,
#         unsmoothed_rv,
#         smoothed_rv,
#         start,
#         stop,
#         _intermediate_step=None,
#         _diffusion=1.0,
#     ):
#         """A single smoother step.
#
#         Consists of predicting from the filtering distribution at time t
#         to time t+1 and then updating based on the discrepancy to the
#         smoothing solution at time t+1.
#
#         Parameters
#         ----------
#         unsmoothed_rv : RandomVariable
#             Filtering distribution at time t.
#         smoothed_rv : RandomVariable
#             Prediction at time t+1 of the filtering distribution at time t.
#         start : float
#             Time-point of the to-be-smoothed RV.
#         stop : float
#             Time-point of the already-smoothed RV.
#         """
#         # It is not clear to me how to best test this, except running IBM smoothing for high-ish order. (N)
#         precon_inv = self.dynamics_model.precon.inverse(stop - start)
#         unsmoothed_rv = precon_inv @ unsmoothed_rv
#         smoothed_rv = precon_inv @ smoothed_rv
#
#         predicted_rv, info = self.dynamics_model.transition_rv_preconditioned(
#             unsmoothed_rv,
#             start,
#             stop=stop,
#             step=_intermediate_step,
#             _diffusion=_diffusion,
#         )
#         crosscov = info["crosscov"]
#         smoothing_gain = np.linalg.solve(predicted_rv.cov.T, crosscov.T).T
#         new_mean = unsmoothed_rv.mean + smoothing_gain @ (
#             smoothed_rv.mean - predicted_rv.mean
#         )
#         new_cov = (
#             unsmoothed_rv.cov
#             + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
#         )
#         return self.dynamics_model.precon(stop - start) @ Normal(new_mean, new_cov)
