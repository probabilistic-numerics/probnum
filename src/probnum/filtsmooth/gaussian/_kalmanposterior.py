"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc
from typing import Iterable, Optional, Union

import numpy as np
from scipy import stats

from probnum import randprocs, randvars, utils
from probnum.filtsmooth import _timeseriesposterior
from probnum.filtsmooth.gaussian import approx
from probnum.typing import (
    DenseOutputLocationArgType,
    FloatArgType,
    IntArgType,
    ShapeArgType,
)

GaussMarkovPriorTransitionArgType = Union[
    randprocs.markov.discrete.DiscreteLinearGaussian,
    approx.DiscreteEKFComponent,
    approx.DiscreteUKFComponent,
    randprocs.markov.continuous.LinearSDE,
    approx.ContinuousEKFComponent,
    approx.ContinuousUKFComponent,
]
"""Any linear and linearised transition can define an (approximate) Gauss-Markov prior."""


class KalmanPosterior(_timeseriesposterior.TimeSeriesPosterior, abc.ABC):
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
        transition: GaussMarkovPriorTransitionArgType,
        locations: Optional[Iterable[FloatArgType]] = None,
        states: Optional[Iterable[randvars.RandomVariable]] = None,
        diffusion_model=None,
    ) -> None:

        super().__init__(locations=locations, states=states)
        self.transition = transition

        self.diffusion_model = diffusion_model
        self.diffusion_model_has_been_provided = diffusion_model is not None

    @abc.abstractmethod
    def interpolate(
        self,
        t: FloatArgType,
        previous_index: Optional[IntArgType] = None,
        next_index: Optional[IntArgType] = None,
    ) -> randvars.RandomVariable:
        raise NotImplementedError

    def sample(
        self,
        rng: np.random.Generator,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
    ) -> np.ndarray:

        size = utils.as_shape(size)
        single_rv_shape = self.states[0].shape
        single_rv_ndim = self.states[0].ndim

        # Early exit if no dense output is required
        if t is None:
            base_measure_realizations = stats.norm.rvs(
                size=(size + self.locations.shape + single_rv_shape),
                random_state=rng,
            )
            return self.transform_base_measure_realizations(
                base_measure_realizations=base_measure_realizations, t=self.locations
            )

        # Compute the union (as sets) of t and self.locations
        # This allows that samples "always pass" the grid points.
        all_locations = np.union1d(t, self.locations)
        slice_these_out = np.where(np.isin(all_locations, t))[0]
        base_measure_realizations = stats.norm.rvs(
            size=(size + all_locations.shape + single_rv_shape),
            random_state=rng,
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
            **Shape (N,).**
            Time points. Must include `self.locations`.Shape

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
        filtering_posterior: _timeseriesposterior.TimeSeriesPosterior,
        transition: GaussMarkovPriorTransitionArgType,
        locations: Iterable[FloatArgType],
        states: Iterable[randvars.RandomVariable],
        diffusion_model=None,
    ):
        self.filtering_posterior = filtering_posterior
        super().__init__(
            transition=transition,
            locations=locations,
            states=states,
            diffusion_model=diffusion_model,
        )

    def interpolate(
        self,
        t: FloatArgType,
        previous_index: Optional[IntArgType] = None,
        next_index: Optional[IntArgType] = None,
    ) -> randvars.RandomVariable:

        # Assert either previous_location or next_location is not None
        # Otherwise, there is no reference point that can be used for interpolation.
        if previous_index is None and next_index is None:
            raise ValueError

        previous_location = (
            self.locations[previous_index] if previous_index is not None else None
        )
        next_location = self.locations[next_index] if next_index is not None else None
        previous_state = (
            self.states[previous_index] if previous_index is not None else None
        )
        next_state = self.states[next_index] if next_index is not None else None

        # Corner case 1: point is on grid. In this case, don't compute anything.
        if t == previous_location:
            return previous_state
        if t == next_location:
            return next_state

        # This block avoids calling self.diffusion_model, because we do not want
        # to search the full index set -- we already know the index!
        # This is the reason that `Diffusion` objects implement a __getitem__.
        # The usual diffusion-index is the next index ('Diffusion's include the right-hand side gridpoint!),
        # but if we are right of the domain, the previous_index matters.
        diffusion_index = next_index if next_index is not None else previous_index
        if diffusion_index >= len(self.locations) - 1:
            diffusion_index = -1
        if self.diffusion_model_has_been_provided:
            squared_diffusion = self.diffusion_model[diffusion_index]
        else:
            squared_diffusion = 1.0

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
            #     next_state, t=next_location, dt=dt, _diffusion=squared_diffusion
            # )
            # return extrapolated_rv_left
            #
            ############################################################

        # Corner case 3: we are extrapolating to the right
        if next_location is None:
            dt = t - previous_location
            assert dt > 0.0
            extrapolated_rv_right, _ = self.transition.forward_rv(
                previous_state, t=previous_location, dt=dt, _diffusion=squared_diffusion
            )
            return extrapolated_rv_right

        # Final case: we are interpolating. Both locations are not None.
        # In this case, filter from the the left to the middle point;
        # And compute a smoothing update from the middle to the RHS point.
        if np.abs(previous_index - next_index) > 1.1:
            raise ValueError
        dt_left = t - previous_location
        dt_right = next_location - t
        assert dt_left > 0.0
        assert dt_right > 0.0
        filtered_rv, _ = self.transition.forward_rv(
            rv=previous_state,
            t=previous_location,
            dt=dt_left,
            _diffusion=squared_diffusion,
        )
        smoothed_rv, _ = self.transition.backward_rv(
            rv_obtained=next_state,
            rv=filtered_rv,
            t=t,
            dt=dt_right,
            _diffusion=squared_diffusion,
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

        # Now we are in the setting of jointly sampling a single realization from the posterior.
        # On time points inside the domain, this is essentially a sequence of smoothing steps.

        t = np.asarray(t) if t is not None else None
        if not np.all(np.isin(self.locations, t)):
            raise ValueError(
                "Base measure realizations cannot be transformed if the locations don't include self.locations."
            )

        if not np.all(np.diff(t) >= 0.0):
            raise ValueError("Time-points have to be sorted.")

        # Find locations of the diffusions, which amounts to finding the locations
        # of the grid points in t (think: `all_locations`), which is done via np.searchsorted:
        diffusion_indices = np.searchsorted(self.locations[:-2], t[1:])
        if self.diffusion_model_has_been_provided:
            squared_diffusion_list = self.diffusion_model[diffusion_indices]
        else:
            squared_diffusion_list = np.ones_like(t)

        # Split into interpolation and extrapolation samples.
        # For extrapolation, samples are propagated forwards.
        # Due to this distinction, we need to treat both cases differently.
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

        squared_diffusion_list_inter = squared_diffusion_list[: len(t_inter)]
        squared_diffusion_list_extra_right = squared_diffusion_list[
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
                _diffusion_list=squared_diffusion_list_inter,
            )
        )
        samples_extra = np.array(
            self.transition.jointly_transform_base_measure_realization_list_forward(
                base_measure_realizations=base_measure_reals_extra_right,
                t=t_extra_right,
                initrv=states_extra_right[0],
                _diffusion_list=squared_diffusion_list_extra_right,
            )
        )
        samples = np.concatenate((samples_inter[:-1], samples_extra), axis=0)
        return samples

    @property
    def _states_left_of_location(self):
        return self.filtering_posterior._states_left_of_location


class FilteringPosterior(KalmanPosterior):
    """Filtering posterior."""

    def interpolate(
        self,
        t: FloatArgType,
        previous_index: Optional[IntArgType] = None,
        next_index: Optional[IntArgType] = None,
    ) -> randvars.RandomVariable:

        # Assert either previous_location or next_location is not None
        # Otherwise, there is no reference point that can be used for interpolation.
        if previous_index is None and next_index is None:
            raise ValueError

        previous_location = (
            self.locations[previous_index] if previous_index is not None else None
        )
        next_location = self.locations[next_index] if next_index is not None else None
        previous_state = (
            self.states[previous_index] if previous_index is not None else None
        )
        next_state = self.states[next_index] if next_index is not None else None

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

        # Final case: we are extrapolating to the right.
        # This is also how the filter-posterior interpolates
        # (by extrapolating from the leftmost point)
        # previous_index is not None
        if self.diffusion_model_has_been_provided:
            diffusion_index = previous_index
            if diffusion_index >= len(self.locations) - 1:
                diffusion_index = -1
            diffusion = self.diffusion_model[diffusion_index]
        else:
            diffusion = 1.0
        dt_left = t - previous_location
        assert dt_left > 0.0
        filtered_rv, _ = self.transition.forward_rv(
            rv=previous_state, t=previous_location, dt=dt_left, _diffusion=diffusion
        )
        return filtered_rv

    def sample(
        self,
        rng: np.random.Generator,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
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
