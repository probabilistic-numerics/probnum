"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import numpy as np

import probnum.random_variables as rvs
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

    def __init__(self, locations, state_rvs, gauss_filter, filter_posterior=None):
        self._locations = np.asarray(locations)
        self.gauss_filter = gauss_filter
        self._state_rvs = _RandomVariableList(state_rvs)

        self.filter_posterior = filter_posterior

    @classmethod
    def from_filterposterior(cls, locations, state_rvs, gauss_filter):
        filter_posterior = cls(
            locations, state_rvs, gauss_filter, filter_posterior=None
        )
        return cls(
            locations=locations,
            state_rvs=[],
            gauss_filter=None,
            filter_posterior=filter_posterior,
        )

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
        # Raise an error if smoothing posterior is called,
        # but only the filtering posterior is available.
        if len(self._locations) == 0:
            raise NotImplementedError("Dense output not available.")

        # Recursive evaluation (t can now be any array, not just length 1)
        if not np.isscalar(t):
            return _RandomVariableList([self.__call__(t_pt) for t_pt in np.asarray(t)])

        # t is left of our grid -- raise error
        # (this functionality is not supported yet)
        if t < self.locations[0]:
            raise ValueError(
                "Invalid location; Can not compute posterior for a location earlier "
                "than the initial location"
            )

        # t is in our grid -- no need to interpolate
        if t in self.locations:
            idx = (self.locations <= t).sum() - 1
            discrete_estimate = self.state_rvs[idx]
            return discrete_estimate

        # Finally: do the interpolation
        if self.locations[0] < t < self.locations[-1]:

            # A) We are the filter posterior:
            # Predict from the left-closest point
            if self.filter_posterior is None:
                return self._predict_to_loc(t)

            # B) We are the smoothing posterior:
            # Predict from the left-closest point and smooth from the right-closest point
            else:
                pred_rv = self.filter_posterior._predict_to_loc(t)
                smoothed_rv = self._smooth_prediction(pred_rv, t)
                return smoothed_rv

        # C) t is beyond the grid, so we predict from either posterior,
        # which yield equivalent results in this case.
        return self._predict_to_loc(t)

    def _predict_to_loc(self, loc):
        """Predict states at location `loc` from the closest, previous state."""
        previous_idx = (self.locations < loc).sum() - 1
        previous_loc = self.locations[previous_idx]
        previous_rv = self.state_rvs[previous_idx]

        pred_rv, _ = self.gauss_filter.predict(
            rv=previous_rv, start=previous_loc, stop=loc
        )
        return pred_rv

    def _smooth_prediction(self, pred_rv, loc):
        """Smooth the predicted state at location `loc` using the next closest."""
        next_idx = (self.locations < loc).sum()
        next_loc = self.locations[next_idx]
        next_rv = self._state_rvs[next_idx]

        # Intermediate prediction
        predicted_future_rv, info = self.gauss_filter.predict(
            rv=pred_rv,
            start=loc,
            stop=next_loc,
        )
        crosscov = info["crosscov"]

        curr_rv, _ = self.gauss_filter.smooth_step(
            pred_rv, predicted_future_rv, next_rv, crosscov
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
            random_vars = self.filter_posterior.state_rvs
        else:
            random_vars = self.filter_posterior(locations)

        if size == ():
            return np.array(
                self._single_sample_path(locations=locations, random_vars=random_vars)
            )

        return np.array(
            [self.sample(locations=locations, size=size[1:]) for _ in range(size[0])]
        )

    def _single_sample_path(self, locations, random_vars):
        # Mirrors gauss_filter.smooth_list, but sample after each step.

        # Make sure the final rv is informed about all the data points.
        # Either condition on the final RV (a single smoothing step) and sample
        # or (if last element in random_vars is final rv) sample directly.
        if locations[-1] < self.locations[-1]:
            t, rv = locations[-1], random_vars[-1]

            # Intermediate prediction
            predicted_rv, info = self.gauss_filter.predict(
                rv=rv,
                start=t,
                stop=self.locations[-1],
            )
            crosscov = info["crosscov"]

            curr_rv, _ = self.gauss_filter.smooth_step(
                rv, predicted_rv, self.state_rvs[-1], crosscov
            )
            curr_sample = curr_rv.sample()
        else:
            curr_rv = random_vars[-1]
            curr_sample = curr_rv.sample()

        # Conclude initialisation
        num_dim = len(curr_sample)
        out_samples = [curr_sample]
        curr_rv = rvs.Normal(curr_sample, np.zeros((num_dim, num_dim)))

        for idx in reversed(range(1, len(locations))):
            unsmoothed_rv = random_vars[idx - 1]

            # Intermediate prediction
            predicted_rv, info = self.gauss_filter.predict(
                rv=unsmoothed_rv,
                start=locations[idx - 1],
                stop=locations[idx],
            )
            crosscov = info["crosscov"]

            curr_rv, _ = self.gauss_filter.smooth_step(
                unsmoothed_rv, predicted_rv, curr_rv, crosscov
            )
            curr_sample = curr_rv.sample()
            out_samples.append(curr_sample)
            curr_rv = rvs.Normal(curr_sample, np.zeros((num_dim, num_dim)))

        out_samples.reverse()
        return out_samples
