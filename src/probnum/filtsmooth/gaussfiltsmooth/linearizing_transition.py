"""Interface for approximate transitions.

Applied to e.g. extended and unscented Kalman filtering and smoothing.
"""

import abc
import typing

import numpy as np

import probnum.random_variables as pnrv
import probnum.type as pntype
from probnum.filtsmooth import statespace


# Naming may be an overfit to EKF, but the next more general name that admits UKF
# would be something like "ApproximateTransition" which says less than LinearizingTransition?
class LinearizingTransition(statespace.Transition, abc.ABC):
    """Approximation of a transition that makes transitioning RVs tractable.

    Joint interface for extended Kalman filtering and unscented Kalman
    filtering.
    """

    def __init__(self, non_linear_model) -> None:
        self.non_linear_model = non_linear_model
        super().__init__()

    @abc.abstractmethod
    def transition_realization(
        self,
        real: np.ndarray,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
    ) -> (pnrv.RandomVariable, typing.Dict):

        raise NotImplementedError

    def transition_realization_preconditioned(
        self,
        real: np.ndarray,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
    ) -> (pnrv.RandomVariable, typing.Dict):
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(
        self,
        rv: pnrv.RandomVariable,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
    ) -> (pnrv.RandomVariable, typing.Dict):
        raise NotImplementedError

    def transition_rv_preconditioned(
        self,
        rv: pnrv.RandomVariable,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
    ) -> (pnrv.RandomVariable, typing.Dict):
        raise NotImplementedError

    @abc.abstractmethod
    def linearize(self, at_this_rv: pnrv.RandomVariable) -> None:
        """Linearize the transition and make it tractable.

        For the EKF, it means assembling the Taylor approximation. For
        the UKF, this means assembling the sigma-points. For general
        quadrature filters, this means assembling the quadrature weights
        and nodes.

        This operation changes self.linearized_model, and does not return anything.
        """
        # No return value, because of semantics:
        # Linearization makes the transition "tractable"
        # but does not change the transition object.
        # In principle, you could linearize once and transition RVs until
        # the end of time. The object would be the same.
        raise NotImplementedError
