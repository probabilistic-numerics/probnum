"""Gaussian filtering and smoothing."""


import numpy as np

import probnum.random_variables as pnrv
from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth import statespace
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.gaussfiltsmooth.kalmanposterior import KalmanPosterior

from .extendedkalman import ContinuousEKFComponent
from .kalman_utils import (
    measure_via_transition,
    predict_via_transition,
    rts_smooth_step_classic,
    update_classic,
)
from .stoppingcriterion import StoppingCriterion
from .unscentedkalman import ContinuousUKFComponent


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
        use_predict=predict_via_transition,
        use_measure=measure_via_transition,
        use_update=update_classic,
        use_smooth_step=rts_smooth_step_classic,
    ):

        if not issubclass(type(initrv), pnrv.Normal):
            raise ValueError(
                "Gaussian filters/smoothers need initial "
                "random variables with Normal distribution."
            )
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv
        self.predict = lambda *args, **kwargs: use_predict(
            dynamics_model, *args, **kwargs
        )
        self.measure = lambda *args, **kwargs: use_measure(
            measurement_model, *args, **kwargs
        )
        self.update = lambda *args, **kwargs: use_update(
            measurement_model, *args, **kwargs
        )
        self.smooth_step = use_smooth_step
        super().__init__(
            dynamics_model=dynamics_model,
            measurement_model=measurement_model,
            initrv=initrv,
        )

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
        new_mean = new_posterior.state_rvs.mean
        old_mean = np.inf * np.ones(new_mean.shape)
        while not stopcrit.terminate(error=new_mean - old_mean, reference=new_mean):
            old_posterior = new_posterior
            new_posterior = self.filtsmooth(
                dataset=dataset,
                times=times,
                _intermediate_step=_intermediate_step,
                _previous_posterior=old_posterior,
            )
            new_mean = new_posterior.state_rvs.mean
            old_mean = old_posterior.state_rvs.mean
        return new_posterior

    def filtsmooth(
        self, dataset, times, _intermediate_step=None, _previous_posterior=None
    ):
        """Apply Gaussian filtering and smoothing to a data set.

        Parameters
        ----------
        dataset : array_like, shape (N, M)
            Data set that is filtered.
        times : array_like, shape (N,)
            Temporal locations of the data points.
            The zeroth element in times and dataset is the location of the initial random variable.
        _intermediate_step
            Step-size to solve the moment equations with. Optional. Only required for LinearSDE priors (not for LTI SDE priors).
        _previous_posterior: KalmanPosterior
            If specified, approximate Gaussian filtering and smoothing linearises at this, prescribed posterior.
            This is used for iterated filtering and smoothing. For standard filtering, this can be ignored.


        Returns
        -------
        KalmanPosterior
            Posterior distribution of the filtered output
        """
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
        """Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        dataset : array_like, shape (N, M)
            Data set that is filtered.
        times : array_like, shape (N,)
            Temporal locations of the data points.
            The zeroth element in times and dataset is the location of the initial random variable.
        _intermediate_step
            Step-size to solve the moment equations with. Optional. Only required for LinearSDE priors (not for LTI SDE priors).
        _previous_posterior: KalmanPosterior
            If specified, approximate Gaussian filtering and smoothing linearises at this, prescribed posterior.
            This is used for iterated filtering and smoothing. For standard filtering, this can be ignored.

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the filtered output
        """
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
        """A single filter step.

        Consists of a prediction step (t -> t+1) and an update step (at t+1).

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
        _intermediate_step
            Intermediate step to be taken inside the prediction.
        _linearise_predict_at
            Linearise the prediction step at this RV. Used for iterated filtering and smoothing.
        _linearise_update_at
            Linearise the update step at this RV. Used for iterated filtering and smoothing.
        _diffusion
            Custom diffusion for the underlying Wiener process. Used in calibration.

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        dict
            Additional information provided by predict() and update().
            Contains keys `pred_rv`, `info_pred`, `meas_rv`, `info_upd`.
        """
        data = np.asarray(data)
        info = {}
        info["pred_rv"], info["info_pred"] = self.predict(
            rv=current_rv,
            start=start,
            stop=stop,
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
        """Apply Gaussian smoothing to the filtering outcome (i.e. a KalmanPosterior).

        Parameters
        ----------
        filter_posterior : KalmanPosterior
            Posterior distribution obtained after filtering

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the smoothed output
        """

        # See Issue #300
        bad_options = (
            statespace.LinearSDE,
            ContinuousEKFComponent,
            ContinuousUKFComponent,
        )
        if isinstance(self.dynamics_model, bad_options) and not isinstance(
            self.dynamics_model, statespace.LTISDE
        ):
            raise ValueError("Continuous-discrete smoothing is not supported (yet).")

        rv_list = self.smooth_list(
            filter_posterior,
            filter_posterior.locations,
        )
        return KalmanPosterior(
            filter_posterior.locations, rv_list, self, with_smoothing=True
        )

    def smooth_list(self, rv_list, locations):
        """Apply smoothing to a list of RVs.

        Parameters
        ----------
        rv_list : _RandomVariableList or array_like
            List of random variables to be smoothed.
        locations : array_like
            Locations of the random variables in rv_list.

        Returns
        -------
        _RandomVariableList
            List of smoothed random variables.
        """

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

            # Actual smoothing step
            curr_rv, _ = self.smooth_step(
                unsmoothed_rv,
                predicted_rv,
                curr_rv,
                crosscov,
                dynamics_model=self.dynamics_model,
                start=locations[idx - 1],
                stop=locations[idx],
            )
            out_rvs.append(curr_rv)
        out_rvs.reverse()
        return _RandomVariableList(out_rvs)
