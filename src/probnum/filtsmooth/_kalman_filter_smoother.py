"""Convenience functions for filtering and smoothing."""

import numpy as np

from probnum import problems, randprocs, randvars
from probnum.filtsmooth import gaussian

__all__ = ["filter_kalman", "smooth_rts"]


def filter_kalman(
    observations, locations, F, L, H, R, m0, C0, prior_model="continuous"
):
    r"""Estimate an unknown, hidden trajectory from a set of observations with a Kalman filter.

    A Kalman filter estimates the unknown trajectory :math:`X` from a set of observations `Y`.
    There is a continuous-discrete and a discrete-discrete version (describing whether the prior model and measurement model are continuous/discrete).

    In a continuous-discrete model, the prior distribution is described by the SDE

    .. math:: \text{d}X(t) = F X(t) \text{d}t + L \text{d}W(t)

    driven by Wiener process :math:`W` and subject to initial condition

    .. math:: X(t_0) \sim N(m_0, C_0).

    By default, :math:`t_0` is set to the location of the first observation.

    In a discrete-discrete model, the prior distribution is described by the transition

    .. math:: X_{n+1} \,|\, X_n \sim N(F X_n, L)

    subject to the same initial condition.

    In both cases, the measurement model is (write :math:`X(t_n)=X_n` in the continuous case)

    .. math:: Y_n \,|\, X_n \sim N(H X_n, R)

    and the Kalman filter estimates :math:`X` given :math:`Y_n=y_n`, :math:`Y=[y_1, ..., y_N]`.

    Parameters
    ----------
    observations
        *(shape=(N, m))* -- A list of noisy observations of the hidden trajectory.
    locations
        *(shape=(N, ))* -- Time-locations of the observations.
    F
        *(shape=(n, n))* -- State transition matrix. Either the drift matrix in an SDE model,
        or the transition matrix in a discrete model (depending on the value of `prior_model`).
    L
        *(shape=(n, n))* or *(shape=(n, s))* -- Diffusion/dispersion matrix. Either the dispersion matrix in an SDE model,
        or the diffusion matrix in a discrete model (depending on the value of `prior_model`).
        In a continuous model, the matrix has shape (n, s) for s-dimensional driving Wiener process.
        In a discrete model, the matrix has shape (n, n).
    H
        *(shape=(m, n))* -- Transition matrix of the (discrete) observation model.
    R
        *(shape=(m, m))* -- Covariance matrix of the observation noise.
    m0
        *(shape=(n,))* -- Initial mean of the prior model.
    C0
        *(shape=(n, n))* -- Initial covariance of the prior model.
    prior_model
        Either discrete (``discrete``) or continuous (``continuous``). This affects the role of `F` and `L`.
        Optional. Default is `continuous`.

    Raises
    ------
    ValueError
        If `prior_model` is neither ``discrete`` nor ``continuous``.

    Returns
    -------
    gaussian.FilteringPosterior
        Filtering distribution as returned by the Kalman filter.
    """
    regression_problem = _setup_regression_problem(
        H=H, R=R, observations=observations, locations=locations
    )
    prior_process = _setup_prior_process(
        F=F, L=L, m0=m0, C0=C0, t0=locations[0], prior_model=prior_model
    )
    kalman = gaussian.Kalman(prior_process)
    return kalman.filter(regression_problem)[0]


def smooth_rts(observations, locations, F, L, H, R, m0, C0, prior_model="continuous"):
    r"""Estimate an unknown, hidden trajectory from a set of observations with a Rauch-Tung-Striebel smoother.

    A Rauch-Tung-Striebel smoother estimates the unknown trajectory :math:`X` from a set of observations `Y`.
    There is a continuous-discrete and a discrete-discrete version (describing whether the prior model and measurement model are continuous/discrete).

    In a continuous-discrete model, the prior distribution is described by the SDE

    .. math:: \text{d}X(t) = F X(t) \text{d}t + L \text{d}W(t)

    driven by Wiener process :math:`W` and subject to initial condition

    .. math:: X(t_0) \sim N(m_0, C_0).

    By default, :math:`t_0` is set to the location of the first observation.

    In a discrete-discrete model, the prior distribution is described by the transition

    .. math:: X_{n+1} \,|\, X_n \sim N(F X_n, L)

    subject to the same initial condition.

    In both cases, the measurement model is (write :math:`X(t_n)=X_n` in the continuous case)

    .. math:: Y_n \,|\, X_n \sim N(H X_n, R)

    and the Rauch-Tung-Striebel smoother estimates :math:`X` given :math:`Y_n=y_n`, :math:`Y=[y_1, ..., y_N]`.

    Parameters
    ----------
    observations
        *(shape=(N, m))* -- A list of noisy observations of the hidden trajectory.
    locations
        *(shape=(N, ))* -- Time-locations of the observations.
    F
        *(shape=(n, n))* -- State transition matrix. Either the drift matrix in an SDE model,
        or the transition matrix in a discrete model (depending on the value of `prior_model`).
    L
        *(shape=(n, n))* or *(shape=(n, s))* -- Diffusion/dispersion matrix. Either the dispersion matrix in an SDE model,
        or the diffusion matrix in a discrete model (depending on the value of `prior_model`).
        In a continuous model, the matrix has shape (n, s) for s-dimensional driving Wiener process.
        In a discrete model, the matrix has shape (n, n).
    H
        *(shape=(m, n))* -- Transition matrix of the (discrete) observation model.
    R
        *(shape=(m, m))* -- Covariance matrix of the observation noise.
    m0
        *(shape=(n,))* -- Initial mean of the prior model.
    C0
        *(shape=(n, n))* -- Initial covariance of the prior model.
    prior_model
        Either discrete (``discrete``) or continuous (``continuous``). This affects the role of `F` and `L`.
        Optional. Default is `continuous`.

    Raises
    ------
    ValueError
        If `prior_model` is neither ``discrete`` nor ``continuous``.

    Returns
    -------
    gaussian.SmoothingPosterior
        Smoothing distribution as returned by the Rauch-Tung-Striebel smoother.
    """
    regression_problem = _setup_regression_problem(
        H=H, R=R, observations=observations, locations=locations
    )
    prior_process = _setup_prior_process(
        F=F, L=L, m0=m0, C0=C0, t0=locations[0], prior_model=prior_model
    )
    kalman = gaussian.Kalman(prior_process)
    return kalman.filtsmooth(regression_problem)[0]


def _setup_prior_process(F, L, m0, C0, t0, prior_model):
    zero_shift_prior = np.zeros(F.shape[0])
    if prior_model == "discrete":
        prior = randprocs.markov.discrete.DiscreteLTIGaussian(
            state_trans_mat=F, shift_vec=zero_shift_prior, proc_noise_cov_mat=L
        )
    elif prior_model == "continuous":
        prior = randprocs.markov.continuous.LTISDE(
            driftmat=F, forcevec=zero_shift_prior, dispmat=L
        )
    else:
        raise ValueError
    initrv = randvars.Normal(m0, C0)
    initarg = t0
    prior_process = randprocs.markov.MarkovProcess(
        transition=prior, initrv=initrv, initarg=initarg
    )
    return prior_process


def _setup_regression_problem(H, R, observations, locations):
    zero_shift_mm = np.zeros(H.shape[0])
    measmod = randprocs.markov.discrete.DiscreteLTIGaussian(
        state_trans_mat=H, shift_vec=zero_shift_mm, proc_noise_cov_mat=R
    )
    measurement_models = [measmod] * len(locations)
    regression_problem = problems.TimeSeriesRegressionProblem(
        observations=observations,
        locations=locations,
        measurement_models=measurement_models,
    )
    return regression_problem
