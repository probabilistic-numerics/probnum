"""Abstract Base Class for posteriors over states after applying filtering/smoothing."""

import abc
from typing import Optional, Union

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.type import (
    ArrayLikeGetitemArgType,
    FloatArgType,
    RandomStateArgType,
    ShapeArgType,
)

DenseOutputLocationArgType = Union[FloatArgType, np.ndarray]
"""TimeSeriesPosteriors and derived classes can be evaluated at a single location 't'
or an array of locations."""

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

    def __init__(self, locations: np.ndarray, states: np.ndarray) -> None:
        self.locations = np.asarray(locations)
        self.states = _randomvariablelist._RandomVariableList(states)

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

        # Recursive evaluation (t can now be any array, not just length 1)
        if not np.isscalar(t):
            return _randomvariablelist._RandomVariableList(
                [self.__call__(t_pt) for t_pt in t]
            )

        # t is left of our grid -- raise error
        # (this functionality is not supported yet)
        if t < self.locations[0]:
            raise ValueError(
                "Invalid location; Can not compute posterior for a location earlier "
                "than the initial location"
            )
        #

        # Early exit if t is in our grid -- no need to interpolate
        if t in self.locations:
            idx = self._find_index(t)
            discrete_estimate = self.states[idx]
            return discrete_estimate

        return self.interpolate(t)

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

    @abc.abstractmethod
    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
    ) -> np.ndarray:
        """Draw samples from the filtering/smoothing posterior.

        If nothing is specified, a single sample is drawn (supported on self.locations).
        If locations are specified, a single sample is drawn on those locations.
        If size is specified, more than a single sample is drawn.

        Internally, samples from a base measure are drawn and transformed via self.transform_base_measure_realizations.

        Parameters
        ----------
        t :
            Locations on which the samples are wanted. Default is none, which implies that
            self.location is used.
        size :
            Indicates how many samples are drawn. Default is an empty tuple, in which case
            a single sample is returned.
        random_state
            Random state (seed, generator) to be used for sampling base measure realizations.


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
        t: Optional[DenseOutputLocationArgType] = None,
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

    def _find_previous_index(self, loc):
        return (self.locations < loc).sum() - 1

    def _find_index(self, loc):
        return self.locations.tolist().index(loc)
