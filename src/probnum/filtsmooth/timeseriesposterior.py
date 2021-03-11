"""Abstract Base Class for posteriors over states after applying filtering/smoothing."""

import abc
from typing import Optional, Union

import numpy as np

from probnum import _randomvariablelist, random_variables
from probnum.type import (
    ArrayLikeGetitemArgType,
    FloatArgType,
    RandomStateArgType,
    ShapeArgType,
)


class TimeSeriesPosterior(abc.ABC):
    """Posterior Distribution over States after time-series algorithms such as
    filtering/smoothing or solving ODEs.

    Parameters
    ----------
    locations
        Locations of the posterior states (represented as random variables).
    state_rvs
        Posterior random variables.
    """

    def __init__(self, locations: np.ndarray, states: np.ndarray) -> None:
        self.locations = np.asarray(locations)
        self.states = _randomvariablelist._RandomVariableList(states)

    def __len__(self) -> int:
        """Length of the discrete-time solution.

        Corresponds to the number of filtering/smoothing steps
        """
        return len(self.locations)

    def __getitem__(
        self, idx: ArrayLikeGetitemArgType
    ) -> random_variables.RandomVariable:
        return self.states[idx]

    @abc.abstractmethod
    def __call__(
        self, t: Union[FloatArgType, np.ndarray]
    ) -> Union[
        random_variables.RandomVariable, _randomvariablelist._RandomVariableList
    ]:
        """Evaluate the time-continuous posterior for a given location.

        Parameters
        ----------
        location : float
            Location, or time, at which to evaluate the posterior.

        Returns
        -------
        rv : `RandomVariable`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self,
        t: Optional[Union[FloatArgType, np.ndarray]] = None,
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

        Returns
        -------
        numpy.ndarray
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
        t: Optional[Union[FloatArgType, np.ndarray]] = None,
        size: Optional[ShapeArgType] = (),
    ):
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
