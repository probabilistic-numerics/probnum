"""Interface for approximate transitions.

Applied to e.g. extended and unscented Kalman filtering and smoothing.
"""

import abc
import typing

import numpy as np

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv


class LinearizingTransition(pnfss.Transition, abc.ABC):
    """Take a non-linear transition and implement an approximation that supports
    transitioning RVs."""

    def __init__(self, non_linear_model: pnfss.Transition):
        self.non_linear_model = non_linear_model
        super().__init__()

    @abc.abstractmethod
    def transition_realization(
        self,
        real: np.ndarray,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        linearise_at: typing.Optional[pnrv.RandomVariable] = None,
    ) -> (pnrv.RandomVariable, typing.Dict):

        raise NotImplementedError

    def transition_realization_preconditioned(
        self,
        real: np.ndarray,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        linearise_at: typing.Optional[pnrv.RandomVariable] = None,
    ) -> (pnrv.RandomVariable, typing.Dict):
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(
        self,
        rv: pnrv.RandomVariable,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        linearise_at: typing.Optional[pnrv.RandomVariable] = None,
    ) -> (pnrv.RandomVariable, typing.Dict):
        raise NotImplementedError

    def transition_rv_preconditioned(
        self,
        rv: pnrv.RandomVariable,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        linearise_at: typing.Optional[pnrv.RandomVariable] = None,
    ) -> (pnrv.RandomVariable, typing.Dict):
        raise NotImplementedError

    @abc.abstractmethod
    def linearize(self, at_this_rv: pnrv.RandomVariable):
        """Linearize the transition and make it tractable.

        For the EKF, it means assembling the Taylor approximation. For
        the UKF, this means assembling the sigma-points. For general
        quadrature filters, this means assembling the quadrature weights
        and nodes.
        """
        raise NotImplementedError
