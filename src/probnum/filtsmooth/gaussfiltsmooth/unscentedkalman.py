"""General Gaussian filters based on approximating intractable quantities with numerical
quadrature.

Examples include the unscented Kalman filter / RTS smoother which is
based on a third degree fully symmetric rule.
"""

import typing

import numpy as np

import probnum.diffeq  # for type annotation in DiscreteEKFComponent.from_ode
import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv
import probnum.type as pntype

from .linearizing_transition import LinearizingTransition
from .unscentedtransform import UnscentedTransform


class UKFComponent(LinearizingTransition):
    """Interface for unscented Kalman filtering components."""

    def __init__(
        self,
        non_linear_model: typing.Union[pnfss.SDE, pnfss.DiscreteGaussian],
        dimension: pntype.IntArgType,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        super().__init__(non_linear_model=non_linear_model)
        self.ut = UnscentedTransform(dimension, spread, priorpar, special_scale)

        # Determine the linearization.
        # Will be constructed later.
        self.sigma_points = None

    def linearize(self, at_this_rv: pnrv.Normal) -> None:
        """Assemble the sigma-points."""
        self.sigma_points = self.ut.sigma_points(at_this_rv.mean, at_this_rv.cov)


class ContinuousUKFComponent(UKFComponent):
    """Continuous unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model: pnfss.SDE,
        dimension: pntype.IntArgType,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        if not isinstance(non_linear_model, pnfss.SDE):
            raise TypeError("cont_model must be an SDE.")
        super().__init__(
            non_linear_model,
            dimension,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

        raise NotImplementedError(
            "Implementation of the continuous UKF is incomplete. It cannot be used."
        )

    def transition_realization(
        self,
        real: np.ndarray,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        raise NotImplementedError("TODO")  # Issue  #234

    def transition_rv(
        self,
        rv: pnrv.Normal,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        raise NotImplementedError("TODO")  # Issue  #234

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class DiscreteUKFComponent(UKFComponent):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model: pnfss.DiscreteGaussian,
        dimension: pntype.IntArgType,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        if not isinstance(non_linear_model, pnfss.DiscreteGaussian):
            raise TypeError("cont_model must be an SDE.")
        super().__init__(
            non_linear_model,
            dimension,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

    def transition_realization(
        self, real: np.ndarray, start: pntype.FloatArgType, _diffusion=1.0, **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        return self.non_linear_model.transition_realization(
            real, start, _diffusion=_diffusion, **kwargs
        )

    def transition_rv(
        self,
        rv: pnrv.Normal,
        start: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        compute_sigmapts_at = _linearise_at if _linearise_at is not None else rv
        self.linearize(at_this_rv=compute_sigmapts_at)

        proppts = self.ut.propagate(
            start, self.sigma_points, self.non_linear_model.state_trans_fun
        )
        meascov = _diffusion * self.non_linear_model.proc_noise_cov_mat_fun(start)
        mean, cov, crosscov = self.ut.estimate_statistics(
            proppts, self.sigma_points, meascov, rv.mean
        )
        return pnrv.Normal(mean, cov), {"crosscov": crosscov}

    @property
    def dimension(self) -> int:
        return self.ut.dimension

    @classmethod
    def from_ode(
        cls,
        ode: "probnum.diffeq.ODE",  # we don't want to import probnum.diffeq here
        prior: pnfss.LinearSDE,
        evlvar: pntype.FloatArgType,
    ) -> "DiscreteUKFComponent":

        spatialdim = prior.spatialdim
        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(spatialdim)

        disc_model = pnfss.DiscreteGaussian(dyna, diff)
        return cls(disc_model, dimension=prior.dimension)
