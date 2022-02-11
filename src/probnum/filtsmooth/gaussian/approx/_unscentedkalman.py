"""Unscented Kalman filtering / spherical cubature Kalman filtering."""

from typing import Optional

import numpy as np
import scipy.linalg

from probnum import randprocs, randvars
from probnum.typing import FloatLike

from ._interface import _LinearizationInterface


class ContinuousUKFComponent(_LinearizationInterface, randprocs.markov.continuous.SDE):
    """Continuous-time unscented Kalman filter transition.

    Parameters
    ----------
    non_linear_model
        Non-linear continuous-time model (:class:`SDE`) that
        is approximated with the UKF.
    mde_atol
        Absolute tolerance passed to the solver
        of the moment differential equations (MDEs). Optional.
    mde_rtol
        Relative tolerance passed to the solver
        of the moment differential equations (MDEs). Optional.
    mde_solver
        Method that is chosen in `scipy.integrate.solve_ivp`.
        Any string that is compatible with
        ``solve_ivp(..., method=mde_solve,...)`` works here.
        Usual candidates are ``[RK45, LSODA, Radau, BDF, RK23, DOP853]``.
        Optional. Default is LSODA.
    """

    def __init__(
        self,
        non_linear_model,
        mde_atol: Optional[FloatLike] = 1e-6,
        mde_rtol: Optional[FloatLike] = 1e-6,
        mde_solver: Optional[str] = "LSODA",
    ) -> None:

        _LinearizationInterface.__init__(self, non_linear_model)
        randprocs.markov.continuous.SDE.__init__(
            self,
            state_dimension=non_linear_model.state_dimension,
            wiener_process_dimension=non_linear_model.wiener_process_dimension,
            drift_function=non_linear_model.drift_function,
            dispersion_function=non_linear_model.dispersion_function,
            drift_jacobian=non_linear_model.drift_jacobian,
        )
        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

        raise NotImplementedError(
            "Implementation of the continuous UKF is incomplete. It cannot be used."
        )


class DiscreteUKFComponent(
    _LinearizationInterface, randprocs.markov.discrete.NonlinearGaussian
):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
    ) -> None:
        _LinearizationInterface.__init__(self, non_linear_model)

        randprocs.markov.discrete.NonlinearGaussian.__init__(
            self,
            input_dim=non_linear_model.input_dim,
            output_dim=non_linear_model.output_dim,
            transition_fun=non_linear_model.transition_fun,
            transition_fun_jacobian=non_linear_model.transition_fun_jacobian,
            noise_fun=non_linear_model.noise_fun,
        )

        self._cubature_params = _spherical_cubature_unit_params(
            dim=non_linear_model.input_dim
        )

    @property
    def dimension(self) -> int:
        """Dimension of the state-space associated with the UKF."""
        return self.ut.dimension

    def linearize(
        self, t, at_this_rv: randvars.RandomVariable
    ) -> randprocs.markov.Transition:
        """Linearize the transition and make it tractable."""
        return _linearize_via_cubature(
            t=t,
            model=self.non_linear_model,
            rv=at_this_rv,
            unit_params=self._cubature_params,
        )


def _spherical_cubature_unit_params(*, dim):
    """Return sigma points and weights for spherical cubature integration.

    Reference:
    Bayesian Filtering and Smoothing. Simo Särkkä. Page 111.
    """
    s, I = np.sqrt(dim), np.eye(dim)
    unit_sigma_points = s * np.concatenate((I, -I), axis=0)
    weights = np.ones(2 * dim) / (2.0 * dim)
    return unit_sigma_points, weights


def _linearize_via_cubature(*, t, model, rv, unit_params):
    """Linearize a nonlinear model statistically with spherical cubature integration."""

    sigma_points_unit, weights = unit_params
    sigma_points = sigma_points_unit @ rv.cov_cholesky.T + rv.mean[None, :]

    sigma_points_transitioned = np.stack(
        [model.transition_fun(t, p) for p in sigma_points], axis=0
    )

    mat, noise_approx = _linearization_system_matrices(
        rv_in=rv,
        weights=weights,
        pts=sigma_points,
        pts_transitioned=sigma_points_transitioned,
    )
    return randprocs.markov.discrete.LinearGaussian(
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        transition_matrix_fun=lambda _: mat,
        noise_fun=lambda s: noise_approx + model.noise_fun(s),
    )


def _linearization_system_matrices(*, rv_in, weights, pts, pts_transitioned):
    """Notation loosely taken from https://arxiv.org/pdf/2102.00514.pdf."""

    pts_centered = pts - rv_in.mean[None, :]
    rv_out, crosscov = _match_moments(
        x_centered=pts_centered, fx=pts_transitioned, weights=weights
    )

    F = scipy.linalg.solve(rv_in.cov, crosscov).T
    mean = rv_out.mean - F @ rv_in.mean
    cov = rv_out.cov - crosscov.T @ F.T
    return F, randvars.Normal(mean=mean, cov=cov)


def _match_moments(*, x_centered, fx, weights):

    fx_mean = weights @ fx
    fx_centered = fx - fx_mean[None, :]

    crosscov = _approx_outer_product(weights, x_centered, fx_centered)
    fx_cov = _approx_outer_product(weights, fx_centered, fx_centered)

    return randvars.Normal(mean=fx_mean, cov=fx_cov), crosscov


def _approx_outer_product(w, a, b):
    outer_product_pt = np.einsum("ijx,ikx->ijk", a[..., None], b[..., None])
    outer_product = np.einsum("i,ijk->jk", w, outer_product_pt)
    return outer_product
