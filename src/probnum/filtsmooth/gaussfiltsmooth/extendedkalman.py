"""Gaussian filtering and smoothing based on making intractable quantities tractable
through Taylor-method approximations, e.g. linearization."""

import abc
import typing

import numpy as np

import probnum.random_variables as pnrv
import probnum.type as pntype
from probnum.filtsmooth import statespace


class EKFComponent(abc.ABC):
    """Interface for extended Kalman filtering components."""

    def __init__(
        self,
        non_linear_model,
    ) -> None:

        self.non_linear_model = non_linear_model

        # Will be constructed later
        self.linearized_model = None

    def forward_realization(
        self,
        real,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ) -> (pnrv.Normal, typing.Dict):

        return self._forward_realization_via_forward_rv(
            real,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ) -> (pnrv.Normal, typing.Dict):

        compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
        self.linearized_model = self.linearize(at_this_rv=compute_jacobian_at)
        return self.linearized_model.forward_rv(
            rv=rv,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
        )

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
        return self._backward_realization_via_backward_rv(
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
        compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
        self.linearized_model = self.linearize(at_this_rv=compute_jacobian_at)
        return self.linearized_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
        )

    @abc.abstractmethod
    def linearize(self, at_this_rv: pnrv.RandomVariable) -> statespace.Transition:
        """Linearize the transition and make it tractable."""
        raise NotImplementedError


# Order of inheritance matters, because forward and backward
# are defined in EKFComponent, and must not be inherited from SDE.
class ContinuousEKFComponent(EKFComponent, statespace.SDE):
    """Continuous extended Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        mde_atol=1e-5,
        mde_rtol=1e-5,
        mde_solver="LSODA",
    ) -> None:

        statespace.SDE.__init__(
            self,
            non_linear_model.driftfun,
            non_linear_model.dispmatfun,
            non_linear_model.jacobfun,
            non_linear_model.dimension,
        )
        EKFComponent.__init__(self, non_linear_model=non_linear_model)

        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

    def linearize(self, at_this_rv: pnrv.Normal):
        """Linearize the drift function with a first order Taylor expansion."""

        g = self.non_linear_model.driftfun
        dg = self.non_linear_model.jacobfun

        x0 = at_this_rv.mean

        def forcevecfun(t):
            return g(t, x0) - dg(t, x0) @ x0

        def driftmatfun(t):
            return dg(t, x0)

        return statespace.LinearSDE(
            dimension=self.non_linear_model.dimension,
            driftmatfun=driftmatfun,
            forcevecfun=forcevecfun,
            dispmatfun=self.non_linear_model.dispmatfun,
            mde_atol=self.mde_atol,
            mde_rtol=self.mde_rtol,
            mde_solver=self.mde_solver,
        )


class DiscreteEKFComponent(EKFComponent, statespace.DiscreteGaussian):
    """Discrete extended Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        forward_implementation="classic",
        backward_implementation="classic",
    ) -> None:

        statespace.DiscreteGaussian.__init__(
            self,
            non_linear_model.input_dim,
            non_linear_model.output_dim,
            non_linear_model.state_trans_fun,
            non_linear_model.proc_noise_cov_mat_fun,
            non_linear_model.jacob_state_trans_fun,
        )
        EKFComponent.__init__(self, non_linear_model=non_linear_model)

        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def linearize(self, at_this_rv: pnrv.Normal):
        """Linearize the dynamics function with a first order Taylor expansion."""

        g = self.non_linear_model.state_trans_fun
        dg = self.non_linear_model.jacob_state_trans_fun

        x0 = at_this_rv.mean

        def forcevecfun(t):
            return g(t, x0) - dg(t, x0) @ x0

        def dynamicsmatfun(t):
            return dg(t, x0)

        return statespace.DiscreteLinearGaussian(
            input_dim=self.non_linear_model.input_dim,
            output_dim=self.non_linear_model.output_dim,
            state_trans_mat_fun=dynamicsmatfun,
            shift_vec_fun=forcevecfun,
            proc_noise_cov_mat_fun=self.non_linear_model.proc_noise_cov_mat_fun,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )

    @classmethod
    def from_ode(
        cls,
        ode,
        prior,
        evlvar,
        ek0_or_ek1=0,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        # code is here, and not in DiscreteGaussian, because we want the option of ek0-Jacobians

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

        discrete_model = statespace.DiscreteGaussian(
            prior.dimension,
            ode.dimension,
            dyna,
            diff,
            jaco,
        )
        return cls(
            discrete_model,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


#
#
# class DiscreteIEKFComponent(DiscreteEKFComponent):
#     """Iterated extended Kalman filtering."""
#
#     def __init__(
#         self,
#         non_linear_model,
#         stopcrit=None,
#         forward_implementation="classic",
#         backward_implementation="classic",
#     ) -> None:
#
#         super().__init__(
#             non_linear_model, forward_implementation, backward_implementation
#         )
#
#         self.stopcrit = StoppingCriterion() if stopcrit is None else stopcrit
#
#     def backward_rv(
#         self,
#         rv_obtained,
#         rv,
#         rv_forwarded=None,
#         gain=None,
#         t=None,
#         dt=None,
#         _diffusion=1.0,
#         _linearise_at=None,
#     ):
#         current_rv, info = self._classic_backward_rv(
#             rv_obtained=rv_obtained,
#             rv=rv,
#             t=t,
#             dt=dt,
#             _diffusion=_diffusion,
#             _linearise_at=_linearise_at,
#         )
#
#         new_mean = current_rv.mean
#         old_mean = np.inf * np.ones(current_rv.mean.shape)
#         while not self.stopcrit.terminate(
#             error=new_mean - old_mean, reference=new_mean
#         ):
#             old_mean = new_mean
#             current_rv, info = self._classic_backward_rv(
#                 rv_obtained=rv_obtained,
#                 rv=current_rv,
#                 t=t,
#                 dt=dt,
#                 _diffusion=_diffusion,
#             )
#             new_mean = current_rv.mean
#         return current_rv, info
#
#     def _classic_backward_rv(
#         self,
#         rv_obtained,
#         rv,
#         t=None,
#         dt=None,
#         _diffusion=1.0,
#         _linearise_at=None,
#     ):
#         compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
#         self.linearized_model = self.linearize(at_this_rv=compute_jacobian_at)
#         return self.linearized_model.backward_rv(
#             rv_obtained=rv_obtained,
#             rv=rv,
#             t=t,
#             dt=dt,
#             _diffusion=_diffusion,
#         )
