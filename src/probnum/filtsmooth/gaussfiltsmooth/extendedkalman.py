"""Gaussian filtering and smoothing based on making intractable quantities tractable
through Taylor-method approximations, e.g. linearization."""

import typing

import numpy as np

import probnum.random_variables as pnrv
import probnum.type as pntype
from probnum.filtsmooth import statespace

from .linearizing_transition import LinearizingTransition


class EKFComponent(LinearizingTransition):
    """Interface for extended Kalman filtering components."""

    def __init__(self, non_linear_model) -> None:
        super().__init__(non_linear_model=non_linear_model)

        # Will be constructed later
        self.linearized_model = None

    def transition_realization(
        self,
        real: np.ndarray,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):

        real_as_rv = pnrv.Normal(real, np.zeros((len(real), len(real))))
        return self.transition_rv(
            real_as_rv,
            start,
            stop,
            step=step,
            _linearise_at=_linearise_at,
            _diffusion=_diffusion,
        )

    def transition_rv(
        self,
        rv: pnrv.Normal,
        start: float,
        stop: typing.Optional[float] = None,
        step: typing.Optional[float] = None,
        _linearise_at: typing.Optional[pnrv.RandomVariable] = None,
        _diffusion: typing.Optional[pntype.FloatArgType] = 1.0,
        **kwargs
    ) -> (pnrv.Normal, typing.Dict):

        compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
        self.linearize(at_this_rv=compute_jacobian_at)
        return self.linearized_model.transition_rv(
            rv=rv, start=start, stop=stop, step=step, _diffusion=_diffusion
        )


class ContinuousEKFComponent(EKFComponent):
    """Continuous extended Kalman filter transition."""

    def __init__(self, non_linear_model, num_steps: pntype.IntArgType) -> None:
        if not isinstance(non_linear_model, statespace.SDE):
            raise TypeError("Continuous EKF transition requires a (non-linear) SDE.")

        super().__init__(non_linear_model=non_linear_model)

        # Number of RK4 steps to solve the ODE dynamics
        # see linear_sde_statics() below
        self.num_steps = num_steps

    def linearize(self, at_this_rv: pnrv.Normal) -> None:
        """Linearize the drift function with a first order Taylor expansion."""

        g = self.non_linear_model.driftfun
        dg = self.non_linear_model.jacobfun

        x0 = at_this_rv.mean

        def forcevecfun(t):
            return g(t, x0) - dg(t, x0) @ x0

        def driftmatfun(t):
            return dg(t, x0)

        self.linearized_model = statespace.LinearSDE(
            driftmatfun=driftmatfun,
            forcevecfun=forcevecfun,
            dispmatfun=self.non_linear_model.dispmatfun,
        )

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteEKFComponent(EKFComponent):
    """Discrete extended Kalman filter transition."""

    def __init__(self, non_linear_model) -> None:
        if not isinstance(non_linear_model, statespace.DiscreteGaussian):
            raise TypeError(
                "Discrete EKF transition requires a (non-linear) discrete Gaussian transition."
            )

        super().__init__(non_linear_model=non_linear_model)

    def linearize(self, at_this_rv: pnrv.Normal) -> None:
        """Linearize the dynamics function with a first order Taylor expansion."""

        g = self.non_linear_model.state_trans_fun
        dg = self.non_linear_model.jacob_state_trans_fun

        x0 = at_this_rv.mean

        def forcevecfun(t):
            return g(t, x0) - dg(t, x0) @ x0

        def dynamicsmatfun(t):
            return dg(t, x0)

        self.linearized_model = statespace.DiscreteLinearGaussian(
            state_trans_mat_fun=dynamicsmatfun,
            shift_vec_fun=forcevecfun,
            proc_noise_cov_mat_fun=self.non_linear_model.proc_noise_cov_mat_fun,
        )

    @property
    def dimension(self):
        raise NotImplementedError

    @classmethod
    def from_ode(
        cls,
        ode,
        prior,
        evlvar,
        ek0_or_ek1=0,
    ):
        # code is here, because we want the option of ek0-jacobians

        spatialdim = prior.spatialdim
        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(spatialdim)

        def jaco_ek1(t, x):
            return h1 - ode.jacobian(t, h0 @ x) @ h0

        def jaco_ek0(t, x):
            return h1

        if ek0_or_ek1 == 0:
            jaco = jaco_ek0
        elif ek0_or_ek1 == 1:
            jaco = jaco_ek1
        else:
            raise TypeError("ek0_or_ek1 must be 0 or 1, resp.")

        discrete_model = statespace.DiscreteGaussian(dyna, diff, jaco)
        return cls(discrete_model)
