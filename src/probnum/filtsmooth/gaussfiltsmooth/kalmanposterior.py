"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc

import numpy as np

from probnum import utils
from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class KalmanPosterior(FiltSmoothPosterior, abc.ABC):
    """Interface for posterior distribution after (extended/unscented) Kalman
    filtering/smoothing.

    Parameters
    ----------
    locations : `array_like`
        Locations / Times of the discrete-time estimates.
    state_rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition : :obj:`Transition`
        Dynamics model used as a prior for the filter.
    """

    def __init__(self, locations, state_rvs, transition):

        self._locations = np.asarray(locations)
        self.transition = transition
        self._state_rvs = _RandomVariableList(state_rvs)

    @property
    def locations(self):
        """:obj:`np.ndarray`: Locations / times of the discrete observations"""
        return self._locations

    @property
    def state_rvs(self):
        """:obj:`list` of :obj:`RandomVariable`: Discrete-time posterior state estimates"""
        return self._state_rvs

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        return self.state_rvs[idx]

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
            discrete_estimate = self.state_rvs[idx]
            return discrete_estimate

        return self.interpolate(t)

    @abc.abstractmethod
    def interpolate(self, t):
        """Evaluate the posterior at a measurement-free point."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, locations=None, size=()):
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
    state_rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition : :obj:`Transition`
        Dynamics model used as a prior for the filter.
    """

    def __init__(self, locations, state_rvs, transition, filtering_posterior):
        self.filtering_posterior = filtering_posterior
        super().__init__(locations, state_rvs, transition)

    def interpolate(self, t):

        pred_rv = self.filtering_posterior.interpolate(t)
        next_idx = self._find_previous_index(t) + 1

        # Early exit if we are extrapolating
        if next_idx >= len(self.locations):
            return pred_rv

        next_t = self.locations[next_idx]
        next_rv = self.state_rvs[next_idx]

        # Actual smoothing step
        curr_rv, _ = self.transition.backward_rv(next_rv, pred_rv, t=t, dt=next_t - t)

        return curr_rv

    def sample(self, locations=None, size=()):
        # In the present setting, only works for sampling from the smoothing posterior.

        size = utils.as_shape(size)

        if locations is None:
            locations = self.locations
            random_vars = self.filtering_posterior.state_rvs
        else:
            random_vars = self.filtering_posterior(locations)

            # Inform the final point in the list about all the data by
            # conditioning on the final state rv
            if locations[-1] < self.locations[-1]:
                final_sample = self.state_rvs[-1].sample()
                random_vars[-1], _ = self.transition.backward_realization(
                    final_sample,
                    random_vars[-1],
                    t=locations[-1],
                    dt=self.locations[-1] - locations[-1],
                )

        if size == ():
            return np.array(
                self.transition.jointly_sample_list_backward(
                    locations=locations, rv_list=random_vars
                )
            )

        return np.array(
            [self.sample(locations=locations, size=size[1:]) for _ in range(size[0])]
        )


class FilteringPosterior(KalmanPosterior):
    """Filtering posterior.

    Parameters
    ----------
    locations : `array_like`
        Locations / Times of the discrete-time estimates.
    state_rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition : :obj:`Transition`
        Dynamics model used as a prior for the filter.
    """

    def interpolate(self, t):
        """Predict to the present point."""
        previous_idx = self._find_previous_index(t)
        previous_t = self.locations[previous_idx]
        previous_rv = self.state_rvs[previous_idx]

        rv, _ = self.transition.forward_rv(previous_rv, t=previous_t, dt=t - previous_t)
        return rv

    def sample(self, locations=None, size=()):
        raise NotImplementedError
