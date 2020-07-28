"""
Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing

Contains the discrete time and function outputs.
Provides dense output by being callable.
Can function values can also be accessed by indexing.
"""
import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.prob.randomvariablelist import _RandomVariableList
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class KalmanPosterior(FiltSmoothPosterior):
    """Posterior Distribution after (Extended/Unscented) Kalman Filtering/Smoothing"""

    def __init__(self, locations, state_rvs, kalman_filter):
        self.kalman_filter = kalman_filter
        self._locations = locations
        self._state_rvs = _RandomVariableList(state_rvs)

    @property
    def locations(self):
        """Locations / times of the discrete observations"""
        return self._locations

    @property
    def state_rvs(self):
        """List of time-discrete posterior estimates over states"""
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
            Smooth the posterior; Opposed to just performing a prediction.

        Returns
        -------
        rv : `RandomVariable`
        """

        if t < self.locations[0]:
            raise ValueError(
                "Invalid location; Can not compute posterior for a location earlier "
                "than the initial location"
            )

        if t in self.locations:
            idx = (self.locations <= t).sum() - 1
            out_rv = self.state_rvs[idx]
            return out_rv
        else:
            prev_idx = (self.locations < t).sum() - 1
            prev_time = self.locations[prev_idx]
            prev_rv = self.state_rvs[prev_idx]

            predicted, _ = self.kalman_filter.predict(
                start=prev_time, stop=t, randvar=prev_rv
            )
            out_rv = predicted

            if smoothed and t > self.locations[-1]:
                next_time = self.locations[prev_idx + 1]
                next_rv = self._state_rvs[prev_idx + 1]
                next_pred, crosscov = self.kalman_filter.predict(
                    start=t, stop=next_time, randvar=predicted
                )

                smoothed = self.kalman_filter.smooth_step(
                    predicted, next_pred, next_rv, crosscov
                )

                out_rv = smoothed

            return out_rv

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        return self.state_rvs[idx]
