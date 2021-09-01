"""Gaussian filtering and smoothing based on making intractable quantities tractable
through Taylor-method approximations, e.g. linearization."""

import abc
from typing import Dict, Tuple

from probnum import randprocs, randvars


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
    def linearize(
        self, at_this_rv: randvars.RandomVariable
    ) -> randprocs.markov.Transition:
        """Linearize the transition and make it tractable."""
        raise NotImplementedError


# Order of inheritance matters, because forward and backward
# are defined in EKFComponent, and must not be inherited from SDE.
class ContinuousEKFComponent(EKFComponent, randprocs.markov.continuous.SDE):
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

        randprocs.markov.continuous.SDE.__init__(
            self,
            state_dimension=non_linear_model.state_dimension,
            wiener_process_dimension=non_linear_model.wiener_process_dimension,
            drift_function=non_linear_model.drift_function,
            dispersion_function=non_linear_model.dispersion_function,
            drift_jacobian=non_linear_model.drift_jacobian,
        )
        EKFComponent.__init__(self, non_linear_model=non_linear_model)

        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

        self.forward_implementation = forward_implementation

    def linearize(self, at_this_rv: randvars.Normal):
        """Linearize the drift function with a first order Taylor expansion."""

        g = self.non_linear_model.drift_function
        dg = self.non_linear_model.drift_jacobian
        l = self.non_linear_model.dispersion_function

        x0 = at_this_rv.mean

        def force_vector_function(t):
            return g(t, x0) - dg(t, x0) @ x0

        def drift_matrix_function(t):
            return dg(t, x0)

        def dispersion_matrix_function(t):
            return l(t, x0)

        return randprocs.markov.continuous.LinearSDE(
            state_dimension=self.non_linear_model.state_dimension,
            wiener_process_dimension=self.non_linear_model.wiener_process_dimension,
            drift_matrix_function=drift_matrix_function,
            force_vector_function=force_vector_function,
            dispersion_matrix_function=dispersion_matrix_function,
            mde_atol=self.mde_atol,
            mde_rtol=self.mde_rtol,
            mde_solver=self.mde_solver,
            forward_implementation=self.forward_implementation,
        )


class DiscreteEKFComponent(EKFComponent, randprocs.markov.discrete.NonlinearGaussian):
    """Discrete extended Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        forward_implementation="classic",
        backward_implementation="classic",
    ) -> None:

        randprocs.markov.discrete.NonlinearGaussian.__init__(
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

        def force_vector_function(t):
            return g(t, x0) - dg(t, x0) @ x0

        def dynamicsmatfun(t):
            return dg(t, x0)

        return randprocs.markov.discrete.LinearGaussian(
            input_dim=self.non_linear_model.input_dim,
            output_dim=self.non_linear_model.output_dim,
            state_trans_mat_fun=dynamicsmatfun,
            shift_vec_fun=force_vector_function,
            proc_noise_cov_mat_fun=self.non_linear_model.proc_noise_cov_mat_fun,
            proc_noise_cov_cholesky_fun=self.non_linear_model.proc_noise_cov_cholesky_fun,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )
