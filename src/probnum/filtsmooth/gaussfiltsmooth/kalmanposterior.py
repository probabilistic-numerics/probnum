"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc

import numpy as np
from scipy import stats

from probnum import utils
from probnum._randomvariablelist import _RandomVariableList

from ..timeseriesposterior import TimeSeriesPosterior


class KalmanPosterior(TimeSeriesPosterior, abc.ABC):
    """Posterior distribution after approximate Gaussian filtering and smoothing.

    Parameters
    ----------
    locs : `array_like`
        Locations / Times of the discrete-time estimates.
    state_rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition : :obj:`Transition`
        Dynamics model used as a prior for the filter.
    """

    def __init__(self, locations, states, transition):

        super().__init__(locations=locations, states=states)
        self.transition = transition

    def __call__(self, t):
        """Evaluate the time-continuous posterior at location `t`

        Algorithm:
        1. Find closest t_prev and t_next, with t_prev < t < t_next
        2. Predict from t_prev to t
        3. (if `self._with_smoothing=True`) Predict from t to t_next
        4. (if `self._with_smoothing=True`) Smooth from t_next to t
        5. Return random variable for time t

        Parameters
        ----------
        t : float
            Location, or time, at which to evaluate the posterior.

        Returns
        -------
        :obj:`RandomVariable`
            Estimate of the states at time ``t``.
        """

        # Recursive evaluation (t can now be any array, not just length 1)
        if not np.isscalar(t):
            return _RandomVariableList([self.__call__(t_pt) for t_pt in t])

        # t is left of our grid -- raise error
        # (this functionality is not supported yet)
        if t < self.locations[0]:
            raise ValueError(
                "Invalid location; Can not compute posterior for a location earlier "
                "than the initial location"
            )

        # Early exit if t is in our grid -- no need to interpolate
        if t in self.locations:
            idx = self._find_index(t)
            discrete_estimate = self.states[idx]
            return discrete_estimate

        return self.interpolate(t)

    @abc.abstractmethod
    def interpolate(self, t):
        """Evaluate the posterior at a measurement-free point."""
        raise NotImplementedError

    def sample(self, t=None, size=(), random_state=None):

        size = utils.as_shape(size)

        if t is None:
            t_shape = (len(self.locations),)
        else:
            t_shape = (len(t) + 1,)
        rv_list_shape = (len(self.filtering_posterior.states[0].mean),)

        base_measure_realizations = stats.norm.rvs(
            size=(size + t_shape + rv_list_shape), random_state=random_state
        )
        return self.transform_base_measure_realizations(
            base_measure_realizations=base_measure_realizations, t=t, size=size
        )

    @abc.abstractmethod
    def transform_base_measure_realizations(
        self, base_measure_realizations, t=None, size=()
    ):
        """Transform samples from a base measure to samples from the KalmanPosterior.

        Here, the base measure is a multivariate standard Normal distribution.

        Parameters
        ----------
        base_measure_realizations
            **Shape (*size, N, d).**
            Samples from a multivariate standard Normal distribution.
            `N` is either the `len(self.locations)` (if `t == None`),
            or `len(t) + 1` (if `t != None`). The reason for the `+1` in the latter
            is that samples at arbitrary locations need to be conditioned on
            a sample at the final time point.
        t
            Times. Optional. If None, samples are drawn at `self.locations`.
        size
            Number of samples to draw. Optional. Default is `size=()`.

        Returns
        -------
        np.ndarray
            **Shape (*size, N, d)**
            Transformed base measure realizations. If the inputs are samples
            from a multivariate standard Normal distribution, the results are
            `size` samples from the Kalman posterior at prescribed locations.
        """
        raise NotImplementedError

    def _find_previous_index(self, loc):
        return (self.locations < loc).sum() - 1

    def _find_index(self, loc):
        return self.locations.tolist().index(loc)


class SmoothingPosterior(KalmanPosterior):
    """Smoothing posterior.

    Parameters
    ----------
    locations : `array_like`
        Locations / Times of the discrete-time estimates.
    states : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition : :obj:`Transition`
        Dynamics model used as a prior for the filter.
    """

    def __init__(self, locations, states, transition, filtering_posterior):
        self.filtering_posterior = filtering_posterior
        super().__init__(locations, states, transition)

    def interpolate(self, t):

        pred_rv = self.filtering_posterior.interpolate(t)
        next_idx = self._find_previous_index(t) + 1

        # Early exit if we are extrapolating
        if next_idx >= len(self.locations):
            return pred_rv

        next_t = self.locations[next_idx]
        next_rv = self.states[next_idx]

        # Actual smoothing step
        curr_rv, _ = self.transition.backward_rv(next_rv, pred_rv, t=t, dt=next_t - t)

        return curr_rv

    def transform_base_measure_realizations(
        self, base_measure_realizations, t=None, size=()
    ):
        size = utils.as_shape(size)
        t = np.asarray(t) if t is not None else None

        # Early exit: recursively compute multiple samples
        # if size is not equal to '()'
        if size != ():
            return np.array(
                [
                    self.transform_base_measure_realizations(
                        base_measure_realizations=base_real,
                        t=t,
                        size=size[1:],
                    )
                    for base_real in base_measure_realizations
                ]
            )

        if t is None:
            t = self.locations
            rv_list = self.filtering_posterior.states
        else:
            rv_list = self.filtering_posterior(t)

            # Inform the final point in the list about all the data by
            # conditioning on the final state rv
            if t[-1] < self.locations[-1]:

                final_rv = self.states[-1]
                if np.isscalar(final_rv.mean):
                    final_sample = (
                        final_rv.mean
                        + final_rv.cov_cholesky * base_measure_realizations[-1]
                    )
                else:
                    final_sample = (
                        final_rv.mean
                        + final_rv.cov_cholesky
                        @ base_measure_realizations[-1].reshape((-1,))
                    )
                rv_list[-1], _ = self.transition.backward_realization(
                    final_sample,
                    rv_list[-1],
                    t=t[-1],
                    dt=self.locations[-1] - t[-1],
                )

        return np.array(
            self.transition.jointly_transform_base_measure_realization_list_backward(
                t=t,
                rv_list=rv_list,
                base_measure_samples=base_measure_realizations,
            )
        )


class FilteringPosterior(KalmanPosterior):
    """Filtering posterior.

    Parameters
    ----------
    locations : `array_like`
        Locations / Times of the discrete-time estimates.
    states : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition : :obj:`Transition`
        Dynamics model used as a prior for the filter.
    """

    def interpolate(self, t):
        """Predict to the present point."""
        previous_idx = self._find_previous_index(t)
        previous_t = self.locations[previous_idx]
        previous_rv = self.states[previous_idx]

        rv, _ = self.transition.forward_rv(previous_rv, t=previous_t, dt=t - previous_t)
        return rv

    def sample(self, t=None, size=()):
        raise NotImplementedError

    def transform_base_measure_realizations(
        self, base_measure_realizations, t=None, size=()
    ):
        raise NotImplementedError("Sampling not implemented.")
