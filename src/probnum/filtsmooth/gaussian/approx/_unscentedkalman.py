"""General Gaussian filters based on approximating intractable quantities with numerical
quadrature.

Examples include the unscented Kalman filter / RTS smoother which is based on a third
degree fully symmetric rule.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.linalg

from probnum import randprocs, randvars
from probnum.filtsmooth.gaussian.approx import _unscentedtransform
from probnum.typing import FloatLike

from ._interface import _LinearizationInterface


class UKFComponent(_LinearizationInterface):
    """Interface for unscented Kalman filtering components."""

    def __init__(
        self,
        non_linear_model,
        spread: Optional[FloatLike] = 1e-4,
        priorpar: Optional[FloatLike] = 2.0,
        special_scale: Optional[FloatLike] = 0.0,
    ) -> None:
        super().__init__(non_linear_model=non_linear_model)

        self.ut = _unscentedtransform.UnscentedTransform(
            non_linear_model.input_dim, spread, priorpar, special_scale
        )

        # Determine the linearization.
        # Will be constructed later.
        self.sigma_points = None

    def assemble_sigma_points(self, at_this_rv: randvars.Normal) -> np.ndarray:
        """Assemble the sigma-points."""
        return self.ut.sigma_points(at_this_rv)

    def linearize(
        self, t, at_this_rv: randvars.RandomVariable
    ) -> randprocs.markov.Transition:
        """Linearize the transition and make it tractable."""
        raise NotImplementedError


class ContinuousUKFComponent(UKFComponent, randprocs.markov.continuous.SDE):
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
        spread: Optional[FloatLike] = 1e-4,
        priorpar: Optional[FloatLike] = 2.0,
        special_scale: Optional[FloatLike] = 0.0,
        mde_atol: Optional[FloatLike] = 1e-6,
        mde_rtol: Optional[FloatLike] = 1e-6,
        mde_solver: Optional[str] = "LSODA",
    ) -> None:

        UKFComponent.__init__(
            self,
            non_linear_model,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )
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


class DiscreteUKFComponent(UKFComponent, randprocs.markov.discrete.NonlinearGaussian):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        spread: Optional[FloatLike] = 1e-4,
        priorpar: Optional[FloatLike] = 2.0,
        special_scale: Optional[FloatLike] = 0.0,
    ) -> None:
        UKFComponent.__init__(
            self,
            non_linear_model,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

        randprocs.markov.discrete.NonlinearGaussian.__init__(
            self,
            input_dim=non_linear_model.input_dim,
            output_dim=non_linear_model.output_dim,
            transition_fun=non_linear_model.transition_fun,
            transition_fun_jacobian=non_linear_model.transition_fun_jacobian,
            noise_fun=non_linear_model.noise_fun,
        )

    @property
    def dimension(self) -> int:
        """Dimension of the state-space associated with the UKF."""
        return self.ut.dimension

    def linearize(
        self, t, at_this_rv: randvars.RandomVariable
    ) -> randprocs.markov.Transition:
        """Linearize the transition and make it tractable."""
        return _spherical_cubature_integration(
            t=t, model=self.non_linear_model, rv0=at_this_rv
        )


def _spherical_cubature_integration(*, t, model, rv0):
    """Linearize a nonlinear model statistically with spherical cubature integration."""

    sigma_points, weights = _spherical_cubature_integration_params(
        rv=rv0, dim=model.input_dim
    )

    sigma_points_transitioned = np.stack(
        [model.transition_fun(t, p) for p in sigma_points], axis=0
    )

    mat, noise_approx = _spherical_cubature_system_matrices(
        rv0=rv0,
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


def _spherical_cubature_integration_params(*, rv, dim):

    unit_sigma_points = np.sqrt(dim) * np.concatenate(
        (
            np.eye(dim),
            -np.eye(dim),
        ),
        axis=0,
    )
    sigma_points = unit_sigma_points @ rv.cov_cholesky.T + rv.mean[None, :]
    weights = np.ones(2 * dim) / (2.0 * dim)
    return sigma_points, weights


def _spherical_cubature_system_matrices(*, rv0, weights, pts, pts_transitioned):
    """Notation from: https://arxiv.org/pdf/2102.00514.pdf."""

    mean_input = rv0.mean  # (d_in,)
    mean_output = weights @ pts_transitioned  # (d_out,)

    centered_input = pts - mean_input[None, :]  # (n, d_in)
    centered_output = pts_transitioned - mean_output[None, :]  # (n, d_out)

    crosscov_pt = np.einsum(
        "ijx,ikx->ijk", centered_input[..., None], centered_output[..., None]
    )  # (n, d_in, d_out)
    crosscov = np.einsum("i,ijk->jk", weights, crosscov_pt)  # (d_in, d_out)

    cov_input = rv0.cov  # (d_in, d_in)
    cov_output_pt = np.einsum(
        "ijx,ikx->ijk", centered_output[..., None], centered_output[..., None]
    )  # (n, d_out, d_out)
    cov_output = np.einsum("i,ijk->jk", weights, cov_output_pt)  # (d_out, d_out)

    gain = scipy.linalg.solve(cov_input, crosscov).T  # (d_in, d_out)
    mean = mean_output - gain @ mean_input  # (d_out,)
    cov = cov_output - crosscov.T @ gain.T  # (d_out, d_out)
    return gain, randvars.Normal(mean=mean, cov=cov)
