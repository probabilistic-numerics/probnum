"""General Gaussian filters based on approximating intractable quantities with numerical
quadrature.

Examples include the unscented Kalman filter / RTS smoother which is
based on a third degree fully symmetric rule.
"""

import typing

import numpy as np

import probnum.statespace as pnss
import probnum.type as pntype
from probnum import randvars

from .unscentedtransform import UnscentedTransform


class UKFComponent:
    """Interface for unscented Kalman filtering components."""

    def __init__(
        self,
        non_linear_model,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        self.non_linear_model = non_linear_model
        self.ut = UnscentedTransform(
            non_linear_model.input_dim, spread, priorpar, special_scale
        )

        # Determine the linearization.
        # Will be constructed later.
        self.sigma_points = None

    def assemble_sigma_points(self, at_this_rv: randvars.Normal) -> np.ndarray:
        """Assemble the sigma-points."""
        return self.ut.sigma_points(at_this_rv)


class ContinuousUKFComponent(UKFComponent, pnss.SDE):
    """Continuous-time unscented Kalman filter transition.

    Parameters
    ----------
    non_linear_model
        Non-linear continuous-time model (:class:`SDE`) that is approximated with the UKF.
    mde_atol
        Absolute tolerance passed to the solver of the moment differential equations (MDEs). Optional. Default is 1e-6.
    mde_rtol
        Relative tolerance passed to the solver of the moment differential equations (MDEs). Optional. Default is 1e-6.
    mde_solver
        Method that is chosen in `scipy.integrate.solve_ivp`. Any string that is compatible with ``solve_ivp(..., method=mde_solve,...)`` works here.
        Usual candidates are ``[RK45, LSODA, Radau, BDF, RK23, DOP853]``. Optional. Default is LSODA.
    """

    def __init__(
        self,
        non_linear_model,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
        mde_atol: typing.Optional[pntype.FloatArgType] = 1e-6,
        mde_rtol: typing.Optional[pntype.FloatArgType] = 1e-6,
        mde_solver: typing.Optional[str] = "LSODA",
    ) -> None:

        UKFComponent.__init__(
            self,
            non_linear_model,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )
        pnss.SDE.__init__(
            self,
            non_linear_model.dimension,
            non_linear_model.driftfun,
            non_linear_model.dispmatfun,
            non_linear_model.jacobfun,
        )
        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

        raise NotImplementedError(
            "Implementation of the continuous UKF is incomplete. It cannot be used."
        )

    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ) -> typing.Tuple[randvars.Normal, typing.Dict]:
        return self._forward_realization_as_rv(
            realization,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def forward_rv(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ) -> typing.Tuple[randvars.Normal, typing.Dict]:
        raise NotImplementedError("TODO")  # Issue  #234

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        return self._backward_realization_as_rv(
            realization_obtained,
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


class DiscreteUKFComponent(UKFComponent, pnss.DiscreteGaussian):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        spread: typing.Optional[pntype.FloatArgType] = 1e-4,
        priorpar: typing.Optional[pntype.FloatArgType] = 2.0,
        special_scale: typing.Optional[pntype.FloatArgType] = 0.0,
    ) -> None:
        UKFComponent.__init__(
            self,
            non_linear_model,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

        pnss.DiscreteGaussian.__init__(
            self,
            non_linear_model.input_dim,
            non_linear_model.output_dim,
            non_linear_model.state_trans_fun,
            non_linear_model.proc_noise_cov_mat_fun,
            non_linear_model.jacob_state_trans_fun,
            non_linear_model.proc_noise_cov_cholesky_fun,
        )

    def forward_rv(
        self, rv, t, compute_gain=False, _diffusion=1.0, _linearise_at=None, **kwargs
    ) -> typing.Tuple[randvars.Normal, typing.Dict]:
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
        if compute_gain:
            gain = crosscov @ np.linalg.inv(cov)
            info["gain"] = gain
        return randvars.Normal(mean, cov), info

    def forward_realization(
        self, realization, t, _diffusion=1.0, _linearise_at=None, **kwargs
    ):

        return self._forward_realization_via_forward_rv(
            realization,
            t=t,
            compute_gain=False,
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
        _diffusion=1.0,
        _linearise_at=None,
        **kwargs
    ):

        # this method is inherited from DiscreteGaussian.
        return self._backward_rv_classic(
            rv_obtained,
            rv,
            rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
            _linearise_at=None,
        )

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        _linearise_at=None,
        **kwargs
    ):

        # this method is inherited from DiscreteGaussian.
        return self._backward_realization_via_backward_rv(
            realization_obtained,
            rv,
            rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    @property
    def dimension(self) -> int:
        return self.ut.dimension

    @classmethod
    def from_ode(
        cls,
        ode,
        prior,
        evlvar=0.0,
    ):

        spatialdim = prior.spatialdim
        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(spatialdim)

        def diff_cholesky(t):
            return np.sqrt(evlvar) * np.eye(spatialdim)

        disc_model = pnss.DiscreteGaussian(
            input_dim=prior.dimension,
            output_dim=prior.spatialdim,
            state_trans_fun=dyna,
            proc_noise_cov_mat_fun=diff,
            proc_noise_cov_cholesky_fun=diff_cholesky,
        )
        return cls(disc_model)
