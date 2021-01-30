"""Gaussian filtering and smoothing."""

import functools as ft

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.gaussfiltsmooth.kalmanposterior import KalmanPosterior
from probnum.random_variables import Normal

# default options at initialisation
from .kalman_utils import (
    measure_via_transition,
    predict_via_transition,
    rts_smooth_step_classic,
    update_classic,
)
from .stoppingcriterion import StoppingCriterion


class Kalman(BayesFiltSmooth):
    """Gaussian filtering and smoothing, i.e. Kalman-like filters and smoothers.

    Parameters
    ----------
    dynamics_model
        Prior dynamics. Usually an LTISDE object or an Integrator, but LinearSDE, ContinuousEKFComponent,
        or ContinuousUKFComponent are also valid. Describes a random process in :math:`K` dimensions.
        If an integrator, `K=spatialdim(ordint+1)` for some spatialdim and ordint.
    measurement_model
        Measurement model. Usually an DiscreteLTIGaussian, but any DiscreteLinearGaussian is acceptable.
        This model maps the `K` dimensional prior state (see above) to the `L` dimensional space in which the observation ''live''.
        For 2-dimensional observations, `L=2`.
        If an DiscreteLTIGaussian, the measurement matrix is :math:`L \\times K` dimensional, the forcevec is `L` dimensional and the meascov is `L \\times L` dimensional.
    initrv
        Initial random variable for the prior. This is a `K` dimensional Gaussian distribution (not `L`, because it belongs to the prior)
    predict
        Custom prediction step. Choose between e.g. classical and square-root implementation. Default is 'predict_via_transition`.
    measure
        Custom measurement step. Choose between e.g. classical and square-root implementation. Default is 'measure_via_transition`.
    update
        Custom update step. Choose between e.g. classical, Joseph form, and square-root implementation. Default is 'update_classic`.
    smooth_step
        Custom update step. Choose between e.g. classical, and square-root implementation. Default is 'rts_smooth_step_classic`.
    """

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

    def iterated_filtsmooth(
        self, dataset, times, stopcrit=None, _intermediate_step=None
    ):
        """Compute an iterated smoothing estimate with repeated posterior linearisation.

        If the extended Kalman filter is used, this yields the IEKS. In
        any case, the result is an approximation to the maximum-a-
        posteriori estimate.
        """

        if stopcrit is None:
            stopcrit = StoppingCriterion()

        # Initialise iterated smoother
        old_posterior = self.filtsmooth(
            dataset=dataset,
            times=times,
            _intermediate_step=_intermediate_step,
            _previous_posterior=None,
        )
        new_posterior = old_posterior
        normalised_error = np.inf

        # Iterate until a fixed point is reached
        while normalised_error > 1:
            old_posterior = new_posterior
            new_posterior = self.filtsmooth(
                dataset=dataset,
                times=times,
                _intermediate_step=_intermediate_step,
                _previous_posterior=old_posterior,
            )
            difference = new_posterior.state_rvs.mean - old_posterior.state_rvs.mean
            reference = old_posterior.state_rvs.mean
            normalised_error = stopcrit.evaluate_error(difference, reference)
        return new_posterior

    def filtsmooth(
        self, dataset, times, _intermediate_step=None, _previous_posterior=None
    ):
        dataset, times = np.asarray(dataset), np.asarray(times)
        filter_posterior = self.filter(
            dataset,
            times,
            _intermediate_step=_intermediate_step,
            _previous_posterior=_previous_posterior,
        )
        smooth_posterior = self.smooth(filter_posterior)
        return smooth_posterior

    def filter(self, dataset, times, _intermediate_step=None, _previous_posterior=None):

        dataset, times = np.asarray(dataset), np.asarray(times)
        rvs = []

        _linearise_update_at = (
            None if _previous_posterior is None else _previous_posterior(times[0])
        )
        filtrv, *_ = self.update(
            rv=self.initrv,
            time=times[0],
            data=dataset[0],
            _linearise_at=_linearise_update_at,
        )

        rvs.append(filtrv)
        for idx in range(1, len(times)):
            _linearise_predict_at = (
                None
                if _previous_posterior is None
                else _previous_posterior(times[idx - 1])
            )
            _linearise_update_at = (
                None if _previous_posterior is None else _previous_posterior(times[idx])
            )

            filtrv, _ = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                current_rv=filtrv,
                data=dataset[idx],
                _linearise_predict_at=_linearise_predict_at,
                _linearise_update_at=_linearise_update_at,
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
        _linearise_predict_at=None,
        _linearise_update_at=None,
        _diffusion=1.0,
    ):
        data = np.asarray(data)
        info = {}
        info["pred_rv"], info["info_pred"] = self.predict(
            start=start,
            stop=stop,
            rv=current_rv,
            _intermediate_step=_intermediate_step,
            _linearise_at=_linearise_predict_at,
            _diffusion=_diffusion,
        )
        filtrv, info["meas_rv"], info["info_upd"] = self.update(
            rv=info["pred_rv"],
            time=stop,
            data=data,
            _linearise_at=_linearise_update_at,
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
