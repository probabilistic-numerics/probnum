"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import numpy as np

from probnum import utils
from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class KalmanPosterior(FiltSmoothPosterior):
    """Posterior Distribution after (Extended/Unscented) Kalman Filtering/Smoothing.

    Parameters
    ----------
    locations : `array_like`
        Locations / Times of the discrete-time estimates.
    state_rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    gauss_filter : :obj:`GaussFiltSmooth`
        Filter/smoother used to compute the discrete-time estimates.
    """

    def __init__(self, locations, state_rvs, transition, with_smoothing):
        self._locations = np.asarray(locations)
        self.transition = transition
        self._state_rvs = _RandomVariableList(state_rvs)
        self._with_smoothing = with_smoothing

    @property
    def locations(self):
        """:obj:`np.ndarray`: Locations / times of the discrete observations"""
        return self._locations

    @property
    def state_rvs(self):
        """:obj:`list` of :obj:`RandomVariable`: Discrete-time posterior state estimates"""
        return self._state_rvs

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
        if not np.isscalar(t):
            # recursive evaluation (t can now be any array, not just length 1!)
            return _RandomVariableList([self.__call__(t_pt) for t_pt in np.asarray(t)])

        if t < self.locations[0]:
            raise ValueError(
                "Invalid location; Can not compute posterior for a location earlier "
                "than the initial location"
            )

        if t in self.locations:
            idx = (self.locations <= t).sum() - 1
            discrete_estimate = self.state_rvs[idx]
            return discrete_estimate

        if self.locations[0] < t < self.locations[-1]:
            pred_rv = self._predict_to_loc(t)
            if self._with_smoothing:
                smoothed_rv = self._smooth_prediction(pred_rv, t)
                return smoothed_rv
            else:
                return pred_rv

        # else: t > self.locations[-1], which case we just predict.
        return self._predict_to_loc(t)

    def _predict_to_loc(self, loc):
        """Predict states at location `loc` from the closest, previous state."""
        previous_idx = (self.locations < loc).sum() - 1
        previous_loc = self.locations[previous_idx]
        previous_rv = self.state_rvs[previous_idx]

        pred_rv, _ = self.transition.forward_rv(
            rv=previous_rv, t=previous_loc, dt=loc - previous_loc
        )
        return pred_rv

    def _smooth_prediction(self, pred_rv, loc):
        """Smooth the predicted state at location `loc` using the next closest."""
        next_idx = (self.locations < loc).sum()
        next_loc = self.locations[next_idx]
        next_rv = self._state_rvs[next_idx]

        # Actual smoothing step
        curr_rv, _ = self.transition.backward_rv(
            next_rv, pred_rv, t=loc, dt=next_loc - loc
        )

        return curr_rv

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        return self.state_rvs[idx]

    def sample(self, locations=None, size=()):

        size = utils.as_shape(size)

        if locations is None:
            locations = self.locations
            random_vars = self.state_rvs
        else:
            random_vars = self.__call__(locations)

        if size == ():
            return np.array(
                self.transition.sample_list(locations=locations, rv_list=random_vars)
            )

        return np.array(
            [self.sample(locations=locations, size=size[1:]) for _ in range(size[0])]
        )
