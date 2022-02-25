"""Unscented Kalman filtering / spherical cubature Kalman filtering."""

import numpy as np
import scipy.linalg

from probnum import randprocs, randvars

from ._interface import _LinearizationInterface


class DiscreteUKFComponent(
    _LinearizationInterface, randprocs.markov.discrete.NonlinearGaussian
):
    """Discrete unscented Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        forward_implementation="classic",
        backward_implementation="classic",
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
        self._forward_implementation = forward_implementation
        self._backward_implementation = backward_implementation

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
            forw_impl=self._forward_implementation,
            backw_impl=self._backward_implementation,
        )


def _spherical_cubature_unit_params(*, dim):
    """Return sigma points and weights for spherical cubature integration.

    Reference:
    Bayesian Filtering and Smoothing. Simo Särkkä. Page 111.
    """
    s, I, zeros = np.sqrt(dim), np.eye(dim), np.zeros((1, dim))
    unit_sigma_points = s * np.concatenate((zeros, I, -I), axis=0)
    weights_mean, weights_cov = _weights(dim)
    return unit_sigma_points, (weights_mean, weights_cov)


def _weights(dim):
    spread, priorpar, special_scale = 1.0, 0.0, 0.0
    scale = spread**2 * (dim + special_scale) - dim

    weights_mean = _weights_mean(dim, scale)
    weights_cov = _weights_cov(dim, priorpar, scale, spread)
    return weights_mean, weights_cov


def _weights_mean(dim, scale):
    mw0 = np.ones(1) * scale / (dim + scale)
    mw = np.ones(2 * dim) / (2.0 * (dim + scale))
    weights_mean = np.hstack((mw0, mw))
    return weights_mean


def _weights_cov(dim, priorpar, scale, spread):
    cw0 = np.ones(1) * scale / (dim + scale) + (1 - spread**2 + priorpar)
    cw = np.ones(2 * dim) / (2.0 * (dim + scale))
    weights_cov = np.hstack((cw0, cw))
    return weights_cov


def _linearize_via_cubature(*, t, model, rv, unit_params, forw_impl, backw_impl):
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

    def new_noise(s):
        noise_model = model.noise_fun(s)
        return noise_model + noise_approx

    return randprocs.markov.discrete.LinearGaussian(
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        transition_matrix_fun=lambda _: mat,
        noise_fun=new_noise,
        forward_implementation=forw_impl,
        backward_implementation=backw_impl,
    )


def _linearization_system_matrices(*, rv_in, weights, pts, pts_transitioned):
    """Notation loosely taken from https://arxiv.org/pdf/2102.00514.pdf."""

    pts_centered = pts - rv_in.mean[None, :]
    rv_out, crosscov = _match_moments(
        x_centered=pts_centered, fx=pts_transitioned, weights=weights
    )

    F = scipy.linalg.solve(
        rv_in.cov + 1e-12 * np.eye(*rv_in.cov.shape), crosscov, assume_a="sym"
    ).T
    mean = rv_out.mean - F @ rv_in.mean
    cov = rv_out.cov - crosscov.T @ F.T
    return F, randvars.Normal(mean=mean, cov=cov)


def _match_moments(*, x_centered, fx, weights):

    weights_mean, weights_cov = weights

    fx_mean = weights_mean @ fx
    fx_centered = fx - fx_mean[None, :]

    crosscov = _approx_outer_product(weights_cov, x_centered, fx_centered)
    fx_cov = _approx_outer_product(weights_cov, fx_centered, fx_centered)

    return randvars.Normal(mean=fx_mean, cov=fx_cov), crosscov


def _approx_outer_product(w, a, b):
    outer_product_pt = np.einsum("ijx,ikx->ijk", a[..., None], b[..., None])
    outer_product = np.einsum("i,ijk->jk", w, outer_product_pt)
    return outer_product
