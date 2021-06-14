"""Gaussian filtering and smoothing based on making intractable quantities tractable
through Taylor-method approximations, e.g. linearization."""

import abc
from typing import Dict, Tuple

import numpy as np

from probnum import problems, randvars, statespace


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
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ) -> Tuple[randvars.Normal, Dict]:

        return self._forward_realization_via_forward_rv(
            realization,
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
    ) -> Tuple[randvars.Normal, Dict]:

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
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        return self._backward_realization_via_backward_rv(
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
    def linearize(self, at_this_rv: randvars.RandomVariable) -> statespace.Transition:
        """Linearize the transition and make it tractable."""
        raise NotImplementedError


# Order of inheritance matters, because forward and backward
# are defined in EKFComponent, and must not be inherited from SDE.
class ContinuousEKFComponent(EKFComponent, statespace.SDE):
    """Continuous-time extended Kalman filter transition.

    Parameters
    ----------
    non_linear_model
        Non-linear continuous-time model (:class:`SDE`) that is approximated with the EKF.
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
        mde_atol=1e-5,
        mde_rtol=1e-5,
        mde_solver="RK45",
        forward_implementation="classic",
    ) -> None:

        statespace.SDE.__init__(
            self,
            driftfun=non_linear_model.driftfun,
            dispmatfun=non_linear_model.dispmatfun,
            jacobfun=non_linear_model.jacobfun,
            dimension=non_linear_model.dimension,
        )
        EKFComponent.__init__(self, non_linear_model=non_linear_model)

        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

        self.forward_implementation = forward_implementation

    def linearize(self, at_this_rv: randvars.Normal):
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
            forward_implementation=self.forward_implementation,
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
            non_linear_model.proc_noise_cov_cholesky_fun,
        )
        EKFComponent.__init__(self, non_linear_model=non_linear_model)

        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def linearize(self, at_this_rv: randvars.Normal):
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
            proc_noise_cov_cholesky_fun=self.non_linear_model.proc_noise_cov_cholesky_fun,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )

    @staticmethod
    def wrap_regression_problem(
        regression_problem,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        """Wrap non-linear measurement models inside a regression problems into EKF
        components.

        Examples
        --------
        >>> import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
        >>> problem, _ = filtsmooth_zoo.pendulum(
        ...     measurement_variance=2.0, timespan=(0.0, 10.0), step=0.5
        ... )
        >>> linearized_problem = DiscreteEKFComponent.wrap_regression_problem(problem)
        >>> print(linearized_problem.measurement_models[0])
        DiscreteEKFComponent(input_dim=2, output_dim=1)
        >>> print(linearized_problem.measurement_models[0].forward_implementation)
        classic
        >>> linearized_problem_sqrt = DiscreteEKFComponent.wrap_regression_problem(problem, forward_implementation="sqrt")
        >>> print(linearized_problem_sqrt.measurement_models[0])
        DiscreteEKFComponent(input_dim=2, output_dim=1)
        >>> print(linearized_problem_sqrt.measurement_models[0].forward_implementation)
        sqrt
        """

        def ekf_lin(
            measmod,
            forw_impl=forward_implementation,
            backw_impl=backward_implementation,
        ):
            return DiscreteEKFComponent(
                non_linear_model=measmod,
                forward_implementation=forw_impl,
                backward_implementation=backw_impl,
            )

        measmods = regression_problem.measurement_models
        new_measmods = [ekf_lin(mm) for mm in measmods]
        return problems.TimeSeriesRegressionProblem(
            locations=regression_problem.locations,
            observations=regression_problem.observations,
            solution=regression_problem.solution,
            measurement_models=new_measmods,
        )

    @classmethod
    def from_ode(
        cls,
        ode,
        prior,
        evlvar=0.0,
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

        def diff_cholesky(t):
            return np.sqrt(evlvar) * np.eye(spatialdim)

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
            diff_cholesky,
        )
        return cls(
            discrete_model,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
