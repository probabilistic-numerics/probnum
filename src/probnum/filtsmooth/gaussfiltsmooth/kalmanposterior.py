"""
Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing

Contains the discrete time and function outputs.
Provides dense output by being callable.
Can function values can also be accessed by indexing.
"""
from warnings import warn

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class KalmanPosterior(FiltSmoothPosterior):
    """
    Posterior Distribution after (Extended/Unscented) Kalman Filtering/Smoothing


    Parameters
    ----------
    locations : `array_like`
        Locations / Times of the discrete-time estimates.
    state_rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time estimates.
    gauss_filter : :obj:`GaussFiltSmooth`
        Filter/smoother used to compute the discrete-time estimates.
    """

    def __init__(self, locations, state_rvs, gauss_filter):
        self._locations = np.asarray(locations)
        self.gauss_filter = gauss_filter
        self._state_rvs = _RandomVariableList(state_rvs)

    @property
    def locations(self):
        """:obj:`np.ndarray`: Locations / times of the discrete observations"""
        return self._locations

    @property
    def state_rvs(self):
        """
        :obj:`list` of :obj:`RandomVariable`: Discrete-time posterior state estimates
        """
        return self._state_rvs

    def __call__(self, t, smoothed=True):
        """
        Evaluate the time-continuous posterior at location `t`

        Algorithm:
        1. Find closest t_prev and t_next, with t_prev < t < t_next
        2. Predict from t_prev to t
        3. (if `smoothed=True`) Predict from t to t_next
        4. (if `smoothed=True`) Smooth from t_next to t
        5. Return random variable for time t

        Parameters
        ----------
        t : float
            Location, or time, at which to evaluate the posterior.
        smoothed : bool, optional
            If ``True`` (default) perform smooth interpolation. If ``False`` perform a
            prediction from the previous location, without smoothing.

        Returns
        -------
        :obj:`RandomVariable`
            Estimate of the states at time ``t``.
        """
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
            if smoothed:
                smoothed_rv = self._smooth_prediction(pred_rv, t)
                return smoothed_rv
            else:
                return pred_rv

        # else: t > self.locations[-1]:
        if smoothed:
            warn("`smoothed=True` is ignored for extrapolation.")
        return self._predict_to_loc(t)

    def _predict_to_loc(self, loc):
        """Predict states at location `loc` from the closest, previous state"""
        prev_idx = (self.locations < loc).sum() - 1
        prev_loc = self.locations[prev_idx]
        prev_rv = self.state_rvs[prev_idx]

        pred_rv, _ = self.gauss_filter.predict(
            start=prev_loc, stop=loc, randvar=prev_rv
        )
        return pred_rv

    def _smooth_prediction(self, pred_rv, loc):
        """Smooth the predicted state at location `loc` using the next closest"""
        next_idx = (self.locations < loc).sum()
        next_loc = self.locations[next_idx]
        next_rv = self._state_rvs[next_idx]
        next_pred, crosscov = self.gauss_filter.predict(
            start=loc, stop=next_loc, randvar=pred_rv
        )
        smoothed_rv = self.gauss_filter.smooth_step(
            pred_rv, next_pred, next_rv, crosscov
        )
        return smoothed_rv

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        return self.state_rvs[idx]
