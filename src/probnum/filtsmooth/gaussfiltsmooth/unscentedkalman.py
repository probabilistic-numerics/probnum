"""General Gaussian filters based on approximating intractable quantities with numerical
quadrature.

Examples include the unscented Kalman filter / RTS smoother which is
based on a third degree fully symmetric rule.
"""

import abc
import typing

import numpy as np

import probnum.random_variables as pnrv
import probnum.type as pntype
from probnum.filtsmooth import statespace

from .unscentedtransform import UnscentedTransform


class UKFComponent(statespace.Transition, abc.ABC):
    """Interface for unscented Kalman filtering components."""

    def __init__(
        self,
        non_linear_model,
        dimension: pntype.IntArgType,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        self.non_linear_model = non_linear_model
        self.ut = UnscentedTransform(dimension, spread, priorpar, special_scale)

        # Determine the linearization.
        # Will be constructed later.
        self.sigma_points = None
        super().__init__()

    def assemble_sigma_points(self, at_this_rv: pnrv.Normal) -> np.ndarray:
        """Assemble the sigma-points."""
        return self.ut.sigma_points(at_this_rv.mean, at_this_rv.cov)


class ContinuousUKFComponent(UKFComponent):
    """Continuous unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        dimension: pntype.IntArgType,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        if not isinstance(non_linear_model, statespace.SDE):
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

    def forward_realization(
        self,
        real: np.ndarray,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        raise NotImplementedError("TODO")  # Issue  #234

    def forward_rv(
        self,
        rv: pnrv.Normal,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        raise NotImplementedError("TODO")  # Issue  #234

    def backward_realization(
        self,
        real,
        rv_past,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ):
        raise NotImplementedError

    def backward_rv(
        self,
        rv_futu,
        rv_past,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class DiscreteUKFComponent(UKFComponent):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        dimension: pntype.IntArgType,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        if not isinstance(non_linear_model, statespace.DiscreteGaussian):
            raise TypeError("cont_model must be an SDE.")
        super().__init__(
            non_linear_model,
            dimension,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

    def forward_realization(
        self, real: np.ndarray, start: pntype.FloatArgType, _diffusion=1.0, **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        return self.non_linear_model.forward_realization(
            real, start, _diffusion=_diffusion, **kwargs
        )

    def forward_rv(
        self,
        rv: pnrv.Normal,
        start: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        compute_sigmapts_at = _linearise_at if _linearise_at is not None else rv
        self.sigma_points = self.assemble_sigma_points(at_this_rv=compute_sigmapts_at)

        proppts = self.ut.propagate(
            start, self.sigma_points, self.non_linear_model.state_trans_fun
        )
        meascov = _diffusion * self.non_linear_model.proc_noise_cov_mat_fun(start)
        mean, cov, crosscov = self.ut.estimate_statistics(
            proppts, self.sigma_points, meascov, rv.mean
        )
        return pnrv.Normal(mean, cov), {"crosscov": crosscov}

    def backward_realization(
        self,
        real,
        rv_past,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ):
        raise NotImplementedError

    def backward_rv(
        self,
        rv_futu,
        rv_past,
        start: pntype.FloatArgType,
        stop: pntype.FloatArgType,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return self.ut.dimension

    @classmethod
    def from_ode(
        cls,
        ode,
        prior,
        evlvar,
    ):

        spatialdim = prior.spatialdim
        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(spatialdim)

        disc_model = statespace.DiscreteGaussian(dyna, diff)
        return cls(disc_model, dimension=prior.dimension)
