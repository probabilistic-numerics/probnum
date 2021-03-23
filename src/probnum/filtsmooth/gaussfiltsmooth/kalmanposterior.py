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
    def interpolate(self, t: FloatArgType) -> randvars.RandomVariable:
        """Evaluate the posterior at a measurement-free point.

        Parameters
        ----------
        t :
            Location to evaluate at.

        Returns
        -------
        randvars.RandomVariable or _randomvariablelist._RandomVariableList
            Dense evaluation.
        """
        raise NotImplementedError

    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
    ) -> np.ndarray:

        # Include the final point if a specific grid is demanded
        # and the rightmost point is left of the rightmost data point.
        # If this is not done, the samples are not from the full posterior.
        if t is None:
            sampling_locs = self.locations
            remove_final_point = False
        elif t[-1] >= self.locations[-1]:
            sampling_locs = t
            remove_final_point = False
        else:
            sampling_locs = np.hstack((t, self.locations[-1]))
            remove_final_point = True

        # Infer desired size of the base measure realizations and create them
        size = utils.as_shape(size)
        single_rv_shape = self.states[0].shape
        base_measure_realizations = stats.norm.rvs(
            size=(size + sampling_locs.shape + single_rv_shape),
            random_state=random_state,
        )

        # Transform samples and return the corresponding values.
        # If the final point was artificially added (see above), remove it again.
        transformed_realizations = self.transform_base_measure_realizations(
            base_measure_realizations=base_measure_realizations, t=sampling_locs
        )
        if remove_final_point:
            return self._remove_final_time_point(transformed_realizations)

        return transformed_realizations

    def _remove_final_time_point(self, transformed_realizations):
        """Remove the transformed sample associated with the final time point from the
        transformed samples.

        Careful with the correct slicing!

        Ignore all the size-related dimensions with the Ellipsis, and ignore the RV-shape-related
        dimension with np.take.
        The line below leaves the last `rv_ndim` dimensions untouched, removes the
        last element from the `rv_ndim+1`th dimension (counted from the back) and
        leaves the former dimensions untouched, too.
        In other words, the transformed realization that corresponds to the added final RV is removed.

        This is extracted into a function not only because it allows more thorough documentation, but we would
        also like to reuse it in the KalmanODESolution.
        """
        rv_ndim = self.states[0].ndim
        return np.take(
            transformed_realizations,
            indices=range(transformed_realizations.shape[-(rv_ndim + 1)] - 1),
            axis=-(rv_ndim + 1),
        )

    @abc.abstractmethod
    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
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

    def interpolate(self, t: DenseOutputLocationArgType) -> randvars.RandomVariable:

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
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
    ) -> np.ndarray:
        t = np.asarray(t) if t is not None else None

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

        # The final location is contained in  't' if this function is called from sample().
        # If `transform_base_measure_realizations()` is called directly from the outside,
        # you better know what you're doing ;)
        rv_list = self.filtering_posterior(t)
        return np.array(
            self.transition.jointly_transform_base_measure_realization_list_backward(
                base_measure_realizations=base_measure_realizations,
                t=t,
                rv_list=rv_list,
            )
        )


class FilteringPosterior(KalmanPosterior):
    """Filtering posterior."""

    def interpolate(self, t: DenseOutputLocationArgType) -> randvars.RandomVariable:
        """Predict to the present point.

        Parameters
        ----------
        t :
            Location to evaluate at.

        Returns
        -------
        randvars.RandomVariable
            Dense evaluation.
        """
        previous_idx = self._find_previous_index(t)
        previous_t = self.locations[previous_idx]
        previous_rv = self.states[previous_idx]

        rv, _ = self.transition.forward_rv(previous_rv, t=previous_t, dt=t - previous_t)
        return rv

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
