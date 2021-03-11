"""Posterior over states after applying (Extended/Unscented) Kalman filtering/smoothing.

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc
from typing import Optional, Union

import numpy as np
from scipy import stats

from probnum import _randomvariablelist, random_variables, statespace, utils
from probnum.type import FloatArgType, RandomStateArgType, ShapeArgType

from ..timeseriesposterior import (
    DenseOutputLocationArgType,
    DenseOutputValueType,
    TimeSeriesPosterior,
)
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
    def interpolate(self, t: FloatArgType) -> random_variables.RandomVariable:
        """Evaluate the posterior at a measurement-free point.

        Parameters
        ----------
        t :
            Location to evaluate at.

        Returns
        -------
        random_variables.RandomVariable or _randomvariablelist._RandomVariableList
            Dense evaluation.
        """
        raise NotImplementedError

    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
    ) -> np.ndarray:

        size = utils.as_shape(size)

        # If self.locations are used, the final RV in the list is informed
        # about all the data. If not, the final data point needs to be
        # included in the joint sampling, hence the (len(t) + 1) below.
        if t is None:
            t_shape = (len(self.locations),)
        else:
            t_shape = (len(t) + 1,)

        rv_list_shape = (len(self.filtering_posterior.states[0].mean),)

        base_measure_realizations = stats.norm.rvs(
            size=(size + t_shape + rv_list_shape), random_state=random_state
        )
        return self.transform_base_measure_realizations(
            base_measure_realizations=base_measure_realizations, t=t, size=size
        )

    @abc.abstractmethod
    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
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
        size :
            Number of samples to draw. Optional. Default is `size=()`.

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

    def interpolate(self, t: DenseOutputLocationArgType) -> DenseOutputValueType:

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
        size: Optional[ShapeArgType] = (),
    ) -> np.ndarray:
        size = utils.as_shape(size)
        t = np.asarray(t) if t is not None else None

        # Early exit: recursively compute multiple samples
        # if size is not equal to '()'
        if size != ():
            return np.array(
                [
                    self.transform_base_measure_realizations(
                        base_measure_realizations=base_real,
                        t=t,
                        size=size[1:],
                    )
                    for base_real in base_measure_realizations
                ]
            )

        if t is None:
            t = self.locations
            rv_list = self.filtering_posterior.states
        else:
            rv_list = self.filtering_posterior(t)

            # Inform the final point in the list about all the data by
            # conditioning on the final state rv
            if t[-1] < self.locations[-1]:

                final_rv = self.states[-1]
                final_sample = (
                    final_rv.mean
                    + final_rv.cov_cholesky
                    @ base_measure_realizations[-1].reshape((-1,))
                )
                rv_list[-1], _ = self.transition.backward_realization(
                    final_sample,
                    rv_list[-1],
                    t=t[-1],
                    dt=self.locations[-1] - t[-1],
                )

        return np.array(
            self.transition.jointly_transform_base_measure_realization_list_backward(
                base_measure_realizations=base_measure_realizations,
                t=t,
                rv_list=rv_list,
            )
        )


class FilteringPosterior(KalmanPosterior):
    """Filtering posterior."""

    def interpolate(self, t: DenseOutputLocationArgType) -> DenseOutputValueType:
        """Predict to the present point.

        Parameters
        ----------
        t :
            Location to evaluate at.

        Returns
        -------
        random_variables.RandomVariable or _randomvariablelist._RandomVariableList
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
        raise NotImplementedError

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
    ) -> np.ndarray:
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
