"""Gaussian filtering and smoothing."""


from typing import Iterable, Optional, Union

import numpy as np

from probnum import problems, randprocs
from probnum.filtsmooth import _bayesfiltsmooth, _timeseriesposterior, optim
from probnum.filtsmooth.gaussian import _kalmanposterior, approx

# Measurement models for a Kalman filter can be all sorts of things:
KalmanSingleMeasurementModelType = Union[
    randprocs.markov.discrete.DiscreteLinearGaussian,
    approx.DiscreteEKFComponent,
    approx.DiscreteUKFComponent,
]
KalmanMeasurementModelArgType = Union[
    KalmanSingleMeasurementModelType, Iterable[KalmanSingleMeasurementModelType]
]


class Kalman(_bayesfiltsmooth.BayesFiltSmooth):
    """Gaussian filtering and smoothing, i.e. Kalman-like filters and smoothers.

    Parameters
    ----------
    prior_process
        Prior Gauss-Markov process. Usually a :class:`MarkovProcess` with a :class:`Normal` initial random variable,
        and an :class:`LTISDE` transition or an :class:`IntegratorTransition`, but :class:`LinearSDE`, :class:`ContinuousEKFComponent`,
        or :class:`ContinuousUKFComponent` are also valid. Describes a random process in :math:`K` dimensions.
        If the transition is an integrator, `K=d*(nu+1)` for some d and nu.
    """

    def iterated_filtsmooth(
        self,
        regression_problem: problems.TimeSeriesRegressionProblem,
        init_posterior: Optional[_kalmanposterior.SmoothingPosterior] = None,
        stopcrit: Optional[optim.StoppingCriterion] = None,
    ):
        """Compute an iterated smoothing estimate with repeated posterior linearisation.

        If the extended Kalman filter is used, this yields the IEKS. In
        any case, the result is an approximation to the maximum-a-
        posteriori estimate.

        Parameters
        ----------
        regression_problem :
            Regression problem.
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
        TimeSeriesRegressionProblem: a regression problem data class
        """

        smoothing_post = init_posterior
        info_dicts = None
        for smoothing_post, info_dicts in self.iterated_filtsmooth_posterior_generator(
            regression_problem, init_posterior, stopcrit
        ):
            pass

        return smoothing_post, info_dicts

    def iterated_filtsmooth_posterior_generator(
        self,
        regression_problem: problems.TimeSeriesRegressionProblem,
        init_posterior: Optional[_kalmanposterior.SmoothingPosterior] = None,
        stopcrit: Optional[optim.StoppingCriterion] = None,
    ):
        """Compute iterated smoothing estimates with repeated posterior linearisation.

        If the extended Kalman filter is used, this yields the IEKS. In
        any case, the result is an approximation to the maximum-a-
        posteriori estimate.

        Parameters
        ----------
        regression_problem :
            Regression problem.
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
        TimeSeriesRegressionProblem: a regression problem data class
        """

        if stopcrit is None:
            stopcrit = optim.StoppingCriterion()

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
        regression_problem: problems.TimeSeriesRegressionProblem,
        _previous_posterior: Optional[_timeseriesposterior.TimeSeriesPosterior] = None,
    ):
        """Apply Gaussian filtering and smoothing to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
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
        TimeSeriesRegressionProblem: a regression problem data class
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
        regression_problem: problems.TimeSeriesRegressionProblem,
        _previous_posterior: Optional[_timeseriesposterior.TimeSeriesPosterior] = None,
    ):
        """Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
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
        TimeSeriesRegressionProblem: a regression problem data class
        """

        posterior = _kalmanposterior.FilteringPosterior(
            transition=self.prior_process.transition
        )
        info_dicts = []

        for t, rv, info in self.filtered_states_generator(
            regression_problem, _previous_posterior
        ):
            posterior.append(location=t, state=rv)
            info_dicts.append(info)

        return posterior, info_dicts

    def filtered_states_generator(
        self,
        regression_problem: problems.TimeSeriesRegressionProblem,
        _previous_posterior: Optional[_timeseriesposterior.TimeSeriesPosterior] = None,
    ):
        """Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
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
        TimeSeriesRegressionProblem: a regression problem data class
        """

        # It is not clear at the moment how to implement this cleanly.
        if not np.all(np.diff(regression_problem.locations) > 0):
            raise ValueError(
                "Gaussian filtering expects sorted, non-repeating time points."
            )

        # Initialise
        t_old = self.prior_process.initarg
        curr_rv = self.prior_process.initrv

        # Iterate over data and measurement models
        for t, data, measmod in regression_problem:

            dt = t - t_old
            info_dict = {}

            # Predict if there is a time-increment
            if dt > 0:
                linearise_predict_at = (
                    None if _previous_posterior is None else _previous_posterior(t_old)
                )
                output = self.prior_process.transition.forward_rv(
                    curr_rv, t, dt=dt, _linearise_at=linearise_predict_at
                )
                curr_rv, info_dict["predict_info"] = output

            # Update (even if there is no increment)
            linearise_update_at = (
                None if _previous_posterior is None else _previous_posterior(t)
            )
            curr_rv, info_dict["update_info"] = measmod.backward_realization(
                realization_obtained=data, rv=curr_rv, _linearise_at=linearise_update_at
            )

            yield t, curr_rv, info_dict
            t_old = t

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
        diffusion_list = np.ones_like(filter_posterior.locations[1:])
        rv_list = self.prior_process.transition.smooth_list(
            filter_posterior.states,
            filter_posterior.locations,
            _diffusion_list=diffusion_list,
        )

        return _kalmanposterior.SmoothingPosterior(
            filtering_posterior=filter_posterior,
            transition=self.prior_process.transition,
            locations=filter_posterior.locations,
            states=rv_list,
        )
