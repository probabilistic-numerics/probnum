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
        input_dim=None,
        output_dim=None,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        self.non_linear_model = non_linear_model
        self.ut = UnscentedTransform(input_dim, spread, priorpar, special_scale)

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
            input_dim=dimension,
            output_dim=dimension,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )
        self.dimension = dimension
        raise NotImplementedError(
            "Implementation of the continuous UKF is incomplete. It cannot be used."
        )

    def forward_realization(
        self, real, t, dt=None, _compute_gain=False, _diffusion=1.0, _linearise_at=None
    ) -> (pnrv.Normal, typing.Dict):
        return self._forward_realization_as_rv(
            real,
            t=t,
            dt=dt,
            _compute_gain=_compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def forward_rv(
        self, rv, t, dt=None, _compute_gain=False, _diffusion=1.0, _linearise_at=None
    ) -> (pnrv.Normal, typing.Dict):
        raise NotImplementedError("TODO")  # Issue  #234

    def backward_realization(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        return self._backward_realization_as_rv(
            real_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        raise NotImplementedError("Not available (yet).")


class DiscreteUKFComponent(UKFComponent):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        input_dim=None,
        output_dim=None,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        if not isinstance(non_linear_model, statespace.DiscreteGaussian):
            raise TypeError("Nonlinear model must be discrete.")
        super().__init__(
            non_linear_model,
            input_dim=input_dim,
            output_dim=output_dim,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )
        self.use_backward_rv = use_backward_rv

    def forward_rv(
        self, rv, t, _compute_gain=False, _diffusion=1.0, _linearise_at=None, **kwargs
    ) -> (pnrv.Normal, typing.Dict):
        compute_sigmapts_at = _linearise_at if _linearise_at is not None else rv
        self.sigma_points = self.assemble_sigma_points(at_this_rv=compute_sigmapts_at)

        proppts = self.ut.propagate(
            t, self.sigma_points, self.non_linear_model.state_trans_fun
        )
        meascov = _diffusion * self.non_linear_model.proc_noise_cov_mat_fun(t)
        mean, cov, crosscov = self.ut.estimate_statistics(
            proppts, self.sigma_points, meascov, rv.mean
        )
        info = {"crosscov": crosscov}
        if _compute_gain:
            gain = crosscov @ np.linalg.inv(cov)
            info["gain"] = gain
        return pnrv.Normal(mean, cov), info

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        _linearise_at=None,
        **kwargs
    ):

        return statespace.backward_rv_classic(
            rv_obtained,
            rv,
            rv_forwarded,
            gain=gain,
            t=t,
            discrete_transition=self,
            _diffusion=_diffusion,
        )

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
