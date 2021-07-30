"""Abstract Base Class for posteriors over states after applying filtering/smoothing."""

import abc
from typing import Iterable, Optional, Union

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.typing import (
    ArrayLikeGetitemArgType,
    DenseOutputLocationArgType,
    FloatArgType,
    IntArgType,
    ShapeArgType,
)

DenseOutputValueType = Union[
    randvars.RandomVariable, _randomvariablelist._RandomVariableList
]
"""Dense evaluation of a TimeSeriesPosterior returns a RandomVariable if evaluated at a single location,
and a _RandomVariableList if evaluated at an array of locations."""


class TimeSeriesPosterior(abc.ABC):
    """Posterior Distribution over States after time-series algorithms such as
    filtering/smoothing or solving ODEs.

    Parameters
    ----------
    locations :
        Locations of the posterior states (represented as random variables).
    states :
        Posterior random variables.
    """

    def __init__(
        self,
        locations: Optional[Iterable[FloatArgType]] = None,
        states: Optional[Iterable[randvars.RandomVariable]] = None,
    ) -> None:
        self._locations = list(locations) if locations is not None else []
        self._states = list(states) if states is not None else []
        self._frozen = False

    def _check_location(self, location: FloatArgType) -> FloatArgType:
        if len(self._locations) > 0 and location <= self._locations[-1]:
            _err_msg = "Locations have to be strictly ascending. "
            _err_msg += f"Received {location} <= {self._locations[-1]}."
            raise ValueError(_err_msg)
        return location

    def append(
        self,
        location: FloatArgType,
        state: randvars.RandomVariable,
    ) -> None:

        if self.frozen:
            raise ValueError("Cannot append to frozen TimeSeriesPosterior object.")

        self._locations.append(self._check_location(location))
        self._states.append(state)

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self):
        return self._frozen

    @property
    def locations(self):
        return np.asarray(self._locations)

    @property
    def states(self):
        return _randomvariablelist._RandomVariableList(self._states)

    def __len__(self) -> int:
        """Length of the discrete-time solution.

        Corresponds to the number of filtering/smoothing steps.
        """
        return len(self.locations)

    def __getitem__(self, idx: ArrayLikeGetitemArgType) -> randvars.RandomVariable:
        return self.states[idx]

    def __call__(self, t: DenseOutputLocationArgType) -> DenseOutputValueType:
        """Evaluate the time-continuous posterior at location `t`

        Algorithm:
        1. Find closest t_prev and t_next, with t_prev < t < t_next
        2. Predict from t_prev to t
        3. (if `self._with_smoothing=True`) Predict from t to t_next
        4. (if `self._with_smoothing=True`) Smooth from t_next to t
        5. Return random variable for time t

        Parameters
        ----------
        t :
            Location, or time, at which to evaluate the posterior.

        Returns
        -------
        randvars.RandomVariable or _randomvariablelist._RandomVariableList
            Estimate of the states at time ``t``.
        """
        # The variable "squeeze_eventually" indicates whether
        # the dimension of the time-array has been promoted
        # (which is there for a well-behaved loop).
        # If this has been the case, the final result needs to be
        # reshaped ("squeezed") accordingly.
        if np.isscalar(t):
            t = np.atleast_1d(t)
            t_has_been_promoted = True
        else:
            t_has_been_promoted = False

        if not np.all(np.diff(t) >= 0.0):
            raise ValueError("Time-points have to be sorted.")

        # Split left-extrapolation, interpolation, right_extrapolation
        t0, tmax = np.amin(self.locations), np.amax(self.locations)
        t_extra_left = t[t < t0]
        t_extra_right = t[t > tmax]
        t_inter = t[(t0 <= t) & (t <= tmax)]

        # Indices of t where they would be inserted
        # into self.locations ("left": right-closest states -- this is the default in searchsorted)
        indices = np.searchsorted(self.locations, t_inter, side="left")
        interpolated_values = [
            self.interpolate(
                t=ti,
                previous_index=previdx,
                next_index=nextidx,
            )
            for ti, previdx, nextidx in zip(
                t_inter,
                indices - 1,
                indices,
            )
        ]
        extrapolated_values_left = [
            self.interpolate(t=ti, previous_index=None, next_index=0)
            for ti in t_extra_left
        ]

        extrapolated_values_right = [
            self.interpolate(t=ti, previous_index=-1, next_index=None)
            for ti in t_extra_right
        ]
        dense_output_values = extrapolated_values_left
        dense_output_values.extend(interpolated_values)
        dense_output_values.extend(extrapolated_values_right)

        if t_has_been_promoted:
            return dense_output_values[0]
        return _randomvariablelist._RandomVariableList(dense_output_values)

    @abc.abstractmethod
    def interpolate(
        self,
        t: FloatArgType,
        previous_index: Optional[IntArgType] = None,
        next_index: Optional[IntArgType] = None,
    ) -> randvars.RandomVariable:
        """Evaluate the posterior at a measurement-free point.

        Returns
        -------
        randvars.RandomVariable or _randomvariablelist._RandomVariableList
            Dense evaluation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
    ) -> np.ndarray:
        """Draw samples from the filtering/smoothing posterior.

        If nothing is specified, a single sample is drawn (supported on self.locations).
        If locations are specified, a single sample is drawn on those locations.
        If size is specified, more than a single sample is drawn.

        Internally, samples from a base measure are drawn and transformed via self.transform_base_measure_realizations.

        Parameters
        ----------
        rng :
            Random number generator.
        t :
            Locations on which the samples are wanted. Default is none, which implies that
            self.location is used.
        size :
            Indicates how many samples are drawn. Default is an empty tuple, in which case
            a single sample is returned.


        Returns
        -------
        np.ndarray
            Drawn samples. If size has shape (A1, ..., Z1), locations have shape (L,),
            and the state space model has shape (A2, ..., Z2), the output has
            shape (A1, ..., Z1, L, A2, ..., Z2).
            For example: size=4, len(locations)=4, dim=3 gives shape (4, 4, 3).
        """
        raise NotImplementedError("Sampling is not implemented.")

    @abc.abstractmethod
    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType],
    ) -> np.ndarray:
        """Transform a set of realizations from a base measure into realizations from
        the posterior.

        Parameters
        ----------
        base_measure_realizations :
            Base measure realizations.
        t :
            Locations on which the transformed realizations shall represent realizations from the posterior.

        Returns
        -------
        np.ndarray
            Transformed realizations.
        """
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
