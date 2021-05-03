"""Gaussian filtering and smoothing."""


from typing import Optional

import numpy as np

from probnum import problems
from probnum.filtsmooth.gaussfiltsmooth import stoppingcriterion

from ..bayesfiltsmooth import BayesFiltSmooth
from ..timeseriesposterior import TimeSeriesPosterior
from .kalmanposterior import FilteringPosterior, SmoothingPosterior
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
    """

    def iterated_filtsmooth(
        self,
        regression_problem: problems.RegressionProblem,
        init_posterior: Optional[SmoothingPosterior] = None,
        stopcrit: Optional[stoppingcriterion.StoppingCriterion] = None,
    ):
        """Compute an iterated smoothing estimate with repeated posterior linearisation.

        If the extended Kalman filter is used, this yields the IEKS. In
        any case, the result is an approximation to the maximum-a-
        posteriori estimate.

        Parameters
        ----------
        regression_problem
        init_posterior
            Initial posterior to linearize at. If not specified, linearizes
            at the prediction random variable.
        stopcrit: StoppingCriterion
            A stopping criterion for iterated filtering.

        Returns
        -------
        SmoothingPosterior

        See Also
        --------
        RegressionProblem: a regression problem data class
        """

        smoothing_post = init_posterior
        info_dicts = None
        for smoothing_post, info_dicts in self.iterated_filtsmooth_posterior_generator(
            regression_problem, smoothing_post, stopcrit
        ):
            pass

        return smoothing_post, info_dicts

    def iterated_filtsmooth_posterior_generator(
        self,
        regression_problem: problems.RegressionProblem,
        init_posterior: Optional[SmoothingPosterior] = None,
        stopcrit: Optional[stoppingcriterion.StoppingCriterion] = None,
    ):
        """Compute iterated smoothing estimates with repeated posterior linearisation.

        If the extended Kalman filter is used, this yields the IEKS. In
        any case, the result is an approximation to the maximum-a-
        posteriori estimate.

        Parameters
        ----------
        regression_problem
        init_posterior
            Initial posterior to linearize at. Defaults to computing a (non-iterated)
            smoothing posterior, which amounts to linearizing at the prediction
            random variable.
        stopcrit: StoppingCriterion
            A stopping criterion for iterated filtering.

        Yields
        ------
        SmoothingPosterior
        info_dicts
            list of dictionaries containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """

        if stopcrit is None:
            stopcrit = StoppingCriterion()

        if init_posterior is None:
            # Initialise iterated smoother
            new_posterior, info_dicts = self.filtsmooth(
                regression_problem,
                _previous_posterior=None,
            )
        else:
            new_posterior = init_posterior
            info_dicts = []

        yield new_posterior, info_dicts
        new_mean = new_posterior.states.mean
        old_mean = np.inf * np.ones(new_mean.shape)
        while not stopcrit.terminate(error=new_mean - old_mean, reference=new_mean):
            old_posterior = new_posterior
            new_posterior, info_dicts = self.filtsmooth(
                regression_problem,
                _previous_posterior=old_posterior,
            )
            yield new_posterior, info_dicts
            new_mean = new_posterior.states.mean
            old_mean = old_posterior.states.mean

    def filtsmooth(
        self,
        regression_problem: problems.RegressionProblem,
        _previous_posterior: Optional[TimeSeriesPosterior] = None,
    ):
        """Apply Gaussian filtering and smoothing to a data set.

        Parameters
        ----------
        regression_problem
        _previous_posterior: KalmanPosterior
            If specified, approximate Gaussian filtering and smoothing linearises at this, prescribed posterior.
            This is used for iterated filtering and smoothing. For standard filtering, this can be ignored.

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the filtered output
        info_dicts
            list of dictionaries containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """
        filter_result = self.filter(
            regression_problem,
            _previous_posterior=_previous_posterior,
        )
        filter_posterior, info_dicts = filter_result
        smooth_posterior = self.smooth(filter_posterior)
        return smooth_posterior, info_dicts

    def filter(
        self,
        regression_problem: problems.RegressionProblem,
        _previous_posterior: Optional[TimeSeriesPosterior] = None,
    ):
        """Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        regression_problem
        _previous_posterior: KalmanPosterior
            If specified, approximate Gaussian filtering and smoothing linearises at this, prescribed posterior.
            This is used for iterated filtering and smoothing. For standard filtering, this can be ignored.

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the filtered output
        info_dicts
            list of dictionaries containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """

        filtered_rvs = []
        info_dicts = []

        for rv, info in self.filtered_states_generator(
            regression_problem, _previous_posterior
        ):
            filtered_rvs.append(rv)
            info_dicts.append(info)

        posterior = FilteringPosterior(
            locations=regression_problem.locations,
            states=filtered_rvs,
            transition=self.dynamics_model,
        )

        return posterior, info_dicts

    def filtered_states_generator(
        self,
        regression_problem: problems.RegressionProblem,
        _previous_posterior: Optional[TimeSeriesPosterior] = None,
    ):
        """Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        regression_problem
        _previous_posterior: KalmanPosterior
            If specified, approximate Gaussian filtering and smoothing linearises at this, prescribed posterior.
            This is used for iterated filtering and smoothing. For standard filtering, this can be ignored.

        Yields
        ------
        filtrv
            Random variable returned from prediction and update of the Kalman filter.
        info_dict
            Dictionary containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """
        dataset, times = regression_problem.observations, regression_problem.locations

        _linearise_update_at = (
            None if _previous_posterior is None else _previous_posterior(times[0])
        )

        info_dict = {"pred_rv": self.initrv, "info_pred": {}}
        filtrv, info_dict["info_upd"] = self.update(
            data=dataset[0],
            rv=self.initrv,
            time=times[0],
            _linearise_at=_linearise_update_at,
        )
        yield filtrv, info_dict

        for idx in range(1, len(times)):
            _linearise_predict_at = (
                None
                if _previous_posterior is None
                else _previous_posterior(times[idx - 1])
            )
            _linearise_update_at = (
                None if _previous_posterior is None else _previous_posterior(times[idx])
            )

            filtrv, info_dict = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                current_rv=filtrv,
                data=dataset[idx],
                _linearise_predict_at=_linearise_predict_at,
                _linearise_update_at=_linearise_update_at,
            )

            yield filtrv, info_dict

    def filter_step(
        self,
        start,
        stop,
        current_rv,
        data,
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
            _linearise_at=_linearise_predict_at,
            _diffusion=_diffusion,
        )

        filtrv, info["info_upd"] = self.update(
            rv=info["pred_rv"],
            time=stop,
            data=data,
            _linearise_at=_linearise_update_at,
        )
        return filtrv, info

    def predict(self, rv, start, stop, _linearise_at=None, _diffusion=1.0):
        return self.dynamics_model.forward_rv(
            rv,
            t=start,
            dt=stop - start,
            _linearise_at=_linearise_at,
            _diffusion=_diffusion,
        )

    # Only here for compatibility reasons, not actually used in filter().
    def measure(self, rv, time, _linearise_at=None):
        return self.measurement_model.forward_rv(
            rv,
            t=time,
            _linearise_at=_linearise_at,
        )

    def update(self, rv, time, data, _linearise_at=None):
        return self.measurement_model.backward_realization(
            data, rv, t=time, _linearise_at=_linearise_at
        )

    def smooth(self, filter_posterior, _previous_posterior=None):
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

        rv_list = self.dynamics_model.smooth_list(
            filter_posterior.states, filter_posterior.locations
        )

        return SmoothingPosterior(
            filter_posterior.locations,
            rv_list,
            self.dynamics_model,
            filtering_posterior=filter_posterior,
        )
