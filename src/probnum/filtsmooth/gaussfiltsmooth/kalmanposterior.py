"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc
from typing import Optional, Union

import numpy as np
from scipy import stats

from probnum import _randomvariablelist, randvars, statespace, utils
from probnum.type import FloatArgType, RandomStateArgType, ShapeArgType

from ..timeseriesposterior import DenseOutputLocationArgType, TimeSeriesPosterior
from .extendedkalman import ContinuousEKFComponent, DiscreteEKFComponent
from .unscentedkalman import ContinuousUKFComponent, DiscreteUKFComponent

GaussMarkovPriorTransitionArgType = Union[
    statespace.DiscreteLinearGaussian,
    DiscreteEKFComponent,
    DiscreteUKFComponent,
    statespace.LinearSDE,
    ContinuousEKFComponent,
    ContinuousUKFComponent,
]
"""Any linear and linearised transition can define an (approximate) Gauss-Markov prior."""


class KalmanPosterior(TimeSeriesPosterior, abc.ABC):
    """Posterior distribution after approximate Gaussian filtering and smoothing.

    Parameters
    ----------
    locations :
        Locations / Times of the discrete-time estimates.
    states :
        Estimated states (in the state-space model view) of the discrete-time estimates.
    transition :
        Dynamics model used as a prior for the filter.
    """

    def __init__(
        self,
        locations: np.ndarray,
        states: _randomvariablelist._RandomVariableList,
        transition: GaussMarkovPriorTransitionArgType,
    ) -> None:

        super().__init__(locations=locations, states=states)
        self.transition = transition

    @abc.abstractmethod
    def interpolate(
        self,
        t: FloatArgType,
        previous_location: Optional[FloatArgType] = None,
        previous_state: Optional[randvars.RandomVariable] = None,
        next_location: Optional[FloatArgType] = None,
        next_state: Optional[randvars.RandomVariable] = None,
    ) -> randvars.RandomVariable:
        raise NotImplementedError

    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
    ) -> np.ndarray:

        size = utils.as_shape(size)
        single_rv_shape = self.states[0].shape
        single_rv_ndim = self.states[0].ndim

        # Early exit if no dense output is required
        if t is None:
            base_measure_realizations = stats.norm.rvs(
                size=(size + self.locations.shape + single_rv_shape),
                random_state=random_state,
            )
            return self.transform_base_measure_realizations(
                base_measure_realizations=base_measure_realizations, t=self.locations
            )

        # Compute the union (as sets) of t and self.locations
        # This allows that samples "always pass" the grid points.
        # W don't want to append the values in `t` to self.locations
        # but instead, sample on self.locations and extract the relevant values.
        all_locations = np.union1d(t, self.locations)
        slice_these_out = np.where(np.isin(all_locations, t))[0]
        base_measure_realizations = stats.norm.rvs(
            size=(size + all_locations.shape + single_rv_shape),
            random_state=random_state,
        )
        samples = self.transform_base_measure_realizations(
            base_measure_realizations=base_measure_realizations, t=all_locations
        )
        new_samples = np.take(
            samples, indices=slice_these_out, axis=-(single_rv_ndim + 1)
        )
        return new_samples

    @abc.abstractmethod
    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: DenseOutputLocationArgType,
    ) -> np.ndarray:
        """Transform samples from a base measure to samples from the KalmanPosterior.

        Here, the base measure is a multivariate standard Normal distribution.

        Parameters
        ----------
        base_measure_realizations :
            **Shape (*size, N, d).**
            Samples from a multivariate standard Normal distribution.
            `N` is either the `len(self.locations)` (if `t == None`),
            or `len(t) + 1` (if `t != None`). The reason for the `+1` in the latter
            is that samples at arbitrary locations need to be conditioned on
            a sample at the final time point.
        t :
            Times. Optional. If None, samples are drawn at `self.locations`.

        Returns
        -------
        np.ndarray
            **Shape (*size, N, d)**
            Transformed base measure realizations. If the inputs are samples
            from a multivariate standard Normal distribution, the results are
            `size` samples from the Kalman posterior at prescribed locations.
        """
        raise NotImplementedError


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
    filtering_posterior :
        Filtering posterior.
    """

    def __init__(
        self,
        locations: np.ndarray,
        states: _randomvariablelist._RandomVariableList,
        transition: GaussMarkovPriorTransitionArgType,
        filtering_posterior: TimeSeriesPosterior,
    ):
        self.filtering_posterior = filtering_posterior
        super().__init__(locations, states, transition)

    def interpolate(
        self,
        t: FloatArgType,
        previous_location: Optional[FloatArgType] = None,
        previous_state: Optional[randvars.RandomVariable] = None,
        next_location: Optional[FloatArgType] = None,
        next_state: Optional[randvars.RandomVariable] = None,
    ) -> randvars.RandomVariable:

        # Assert either previous_location or next_location is not None
        if previous_location is None and next_location is None:
            raise ValueError

        # Corner case 1: point is on grid
        if t == previous_location:
            return previous_state
        if t == next_location:
            return next_state

        # Corner case 2: are extrapolating to the left
        if previous_location is None:
            raise NotImplementedError("Extrapolation to the left is not implemented.")
            # The code below would more or less work, but since forward and backward transitions
            # cannot handle negative time increments reliably, we do not support it.
            #
            ############################################################
            #
            # dt = t - next_location
            # assert dt < 0.0
            # extrapolated_rv_left, _ = self.transition.forward_rv(
            #     next_state, t=next_location, dt=dt
            # )
            # return extrapolated_rv_left
            #
            ############################################################

        # Corner case 3: we are extrapolating to the right
        if next_location is None:
            dt = t - previous_location
            assert dt > 0.0
            extrapolated_rv_right, _ = self.transition.forward_rv(
                previous_state, t=previous_location, dt=dt
            )
            return extrapolated_rv_right

        # Final case: we are interpolating. Both locations are not None.
        dt_left = t - previous_location
        dt_right = next_location - t
        assert dt_left > 0.0
        assert dt_right > 0.0
        filtered_rv, _ = self.transition.forward_rv(
            rv=previous_state, t=previous_location, dt=dt_left
        )
        smoothed_rv, _ = self.transition.backward_rv(
            rv_obtained=next_state, rv=filtered_rv, t=t, dt=dt_right
        )
        return smoothed_rv

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t,
    ) -> np.ndarray:

        # Early exit: recursively compute multiple samples
        # if the desired sample size is not equal to '()', which is the case if
        # the shape of base_measure_realization is not (len(locations), shape(RV))
        # t_shape = self.locations.shape if t is None else (len(t) + 1,)
        size_zero_shape = () + t.shape + self.states[0].shape
        if base_measure_realizations.shape != size_zero_shape:
            return np.array(
                [
                    self.transform_base_measure_realizations(
                        base_measure_realizations=base_real,
                        t=t,
                    )
                    for base_real in base_measure_realizations
                ]
            )

        # Now we are in the setting of computing a single sample.
        # The sample is computed by jointly sampling from the posterior.
        # On time points inside the domain, this is essentially a sequence of smoothing steps.
        # For extrapolation, samples are propagated forwards.
        t = np.asarray(t) if t is not None else None
        if not np.all(np.isin(self.locations, t)):
            raise ValueError(
                "Base measure realizations cannot be transformed if the locations don't include self.locations."
            )

        if not np.all(np.diff(t) >= 0.0):
            raise ValueError("Time-points have to be sorted.")

        # Split into interpolation and extrapolation samples.
        # Note: t=tmax is in two arrays!
        # This is on purpose, because sample realisations need to be
        # "communicated" between interpolation and extrapolation.
        t0, tmax = np.amin(self.locations), np.amax(self.locations)
        t_extra_left = t[t < t0]
        t_extra_right = t[tmax <= t]
        t_inter = t[(t0 <= t) & (t <= tmax)]

        if len(t_extra_left) > 0:
            raise NotImplementedError(
                "Sampling on the left of the time-domain is not implemented."
            )

        # Split base measure realisations (which have, say, length N + M - 1):
        # the first N realizations belong to the interpolation samples,
        # and the final M realizations belong to the extrapolation samples.
        # Note again: the sample corresponding to tmax belongs to both groups.
        base_measure_reals_inter = base_measure_realizations[: len(t_inter)]
        base_measure_reals_extra_right = base_measure_realizations[
            -len(t_extra_right) :
        ]

        states = self.filtering_posterior(t)
        states_inter = states[: len(t_inter)]
        states_extra_right = states[-len(t_extra_right) :]

        samples_inter = np.array(
            self.transition.jointly_transform_base_measure_realization_list_backward(
                base_measure_realizations=base_measure_reals_inter,
                t=t_inter,
                rv_list=states_inter,
            )
        )

        samples_extra = np.array(
            self.transition.jointly_transform_base_measure_realization_list_forward(
                base_measure_realizations=base_measure_reals_extra_right,
                t=t_extra_right,
                initrv=states_extra_right[0],
            )
        )
        samples = np.concatenate((samples_inter[:-1], samples_extra), axis=0)

        # if squeeze_eventually:
        #     return samples[0]
        return samples

        #
        #
        # # Split left-extrapolation, interpolation, right_extrapolation
        # t0, tmax = np.amin(self.locations), np.amax(self.locations)
        # t_extra_left = t[t < t0]
        # t_extra_right = t[t > tmax]
        # t_inter = t[(t0 <= t) & (t <= tmax)]
        #
        # # Indices of t where they would be inserted
        # # into self.locations ("left": right-closest states)
        # indices = np.searchsorted(self.locations, t_inter, side="left")
        #
        # # The final location is contained in  't' if this function is called from sample().
        # # If `transform_base_measure_realizations()` is called directly from the outside,
        # # you better know what you're doing ;)
        # rv_list = self.filtering_posterior(t)
        # return np.array(
        #     self.transition.jointly_transform_base_measure_realization_list_backward(
        #         base_measure_realizations=base_measure_realizations,
        #         t=t,
        #         rv_list=rv_list,
        #     )
        # )

    @property
    def _states_left_of_location(self):
        return self.filtering_posterior._states_left_of_location


class FilteringPosterior(KalmanPosterior):
    """Filtering posterior."""

    def interpolate(
        self,
        t: FloatArgType,
        previous_location: Optional[FloatArgType] = None,
        previous_state: Optional[randvars.RandomVariable] = None,
        next_location: Optional[FloatArgType] = None,
        next_state: Optional[randvars.RandomVariable] = None,
    ) -> randvars.RandomVariable:

        # Assert either previous_location or next_location is not None
        if previous_location is None and next_location is None:
            raise ValueError

        # Corner case 1: point is on grid
        if t == previous_location:
            return previous_state
        if t == next_location:
            return next_state

        # Corner case 2: are extrapolating to the left
        if previous_location is None:
            raise NotImplementedError("Extrapolation to the left is not implemented.")
            # The code below would work, but since forward and backward transitions
            # cannot handle negative time increments reliably, we do not support it.
            #
            ############################################################
            #
            # dt = t - next_location
            # assert dt < 0.0
            # extrapolated_rv_left, _ = self.transition.forward_rv(
            #     next_state, t=next_location, dt=dt
            # )
            # return extrapolated_rv_left
            #
            ############################################################

        # Corner case 3: are extrapolating to the right
        if next_location is None:
            dt = t - previous_location
            assert dt > 0.0
            extrapolated_rv_right, _ = self.transition.forward_rv(
                previous_state, t=previous_location, dt=dt
            )
            return extrapolated_rv_right

        # Final case: we are interpolating. Both locations are not None.
        dt_left = t - previous_location
        assert dt_left > 0.0
        filtered_rv, _ = self.transition.forward_rv(
            rv=previous_state, t=previous_location, dt=dt_left
        )
        return filtered_rv

    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
    ) -> np.ndarray:
        # If this error would not be thrown here, trying to sample from a FilteringPosterior
        # would call FilteringPosterior.transform_base_measure_realizations which is not implemented.
        # Since an error thrown by that function instead of one thrown by FilteringPosterior.sample
        # would likely by hard to parse by a user, we explicitly raise a NotImplementedError here.
        raise NotImplementedError(
            "Sampling from the FilteringPosterior is not implemented."
        )

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
