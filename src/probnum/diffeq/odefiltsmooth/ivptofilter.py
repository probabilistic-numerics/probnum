"""
Adapter methods:
from initial value problems + state space model to filters.
"""

import numpy as np
from probnum.filtsmooth import ExtendedKalmanFilter, UnscentedKalmanFilter
from probnum.filtsmooth.statespace.discrete import DiscreteGaussianModel
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal


def ivp_to_ekf0(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for KF based on IVP and prior.

    Returns ExtendedKalmanFilter object that is compatible with
    the GaussianIVPFilter.

    evlvar : float,
        measurement variance; in the literature, this is "R"
    """
    measmod = _measmod_ekf0(ivp, prior, evlvar)
    initrv = _initialdistribution(ivp, prior)
    return ExtendedKalmanFilter(prior, measmod, initrv)


def _measmod_ekf0(ivp, prior, evlvar):
    """
    Zero-th order Taylor approximation as linearisation.

    We call it Kalman filter for convenience;
    it is no Kalman filter in reality.
    """
    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x, **kwargs):
        return h1 @ x - ivp.rhs(t, h0 @ x)

    def diff(t, **kwargs):
        return evlvar * np.eye(spatialdim)

    def jaco(t, x, **kwargs):
        return h1

    return DiscreteGaussianModel(dyna, diff, jaco)


def ivp_to_ekf1(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for EKF based on IVP and prior.

    Returns ExtendedKalmanFilter object.

    evlvar : float, (this is "R")
    """
    measmod = _measmod_ekf1(ivp, prior, evlvar)
    initrv = _initialdistribution(ivp, prior)
    return ExtendedKalmanFilter(prior, measmod, initrv)


def _measmod_ekf1(ivp, prior, evlvar):
    """
    Computes H and R
    """
    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x, **kwargs):
        return h1 @ x - ivp.rhs(t, h0 @ x)

    def diff(t, **kwargs):
        return evlvar * np.eye(spatialdim)

    def jaco(t, x, **kwargs):
        return h1 - ivp.jacobian(t, h0 @ x) @ h0

    return DiscreteGaussianModel(dyna, diff, jaco)


def ivp_to_ukf(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for EKF based on IVP and prior.

    Returns ExtendedKalmanFilter object.

    evlvar : float, (this is "R")
    """
    measmod = _measmod_ukf(ivp, prior, evlvar)
    initrv = _initialdistribution(ivp, prior)
    return UnscentedKalmanFilter(prior, measmod,
                                 initrv, 1.0, 1.0, 1.0)


def _measmod_ukf(ivp, prior, measvar):
    """
    """
    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x, **kwargs):
        return h1 @ x - ivp.rhs(t, h0 @ x)

    def diff(t, **kwargs):
        return measvar * np.eye(spatialdim)

    return DiscreteGaussianModel(dyna, diff)


def _initialdistribution(ivp, prior):
    """
    Perform initial Kalman update to condition the initial distribution
    of the prior on the initial values (and all available derivatives).
    """
    x0 = ivp.initialdistribution.mean()
    dx0 = ivp.rhs(ivp.t0, x0)
    ddx0 = _ddx(ivp.t0, x0, ivp)

    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)
    if prior.ordint == 1:
        projmtrx = np.hstack((h0.T, h1.T)).T
        data = np.hstack((x0, dx0))
    else:
        h2 = prior.proj2coord(coord=2)
        projmtrx = np.hstack((h0.T, h1.T, h2.T)).T
        data = np.hstack((x0, dx0, ddx0))
    s = projmtrx @ projmtrx.T
    newmean = projmtrx.T @ np.linalg.solve(s, data)
    newcov = np.eye(len(newmean)) - projmtrx.T @ np.linalg.solve(s, projmtrx)
    return RandomVariable(distribution=Normal(newmean, newcov))


def _ddx(t, x, ivp):
    """
    If Jacobian is available:
    x''(t) = J_f(x(t)) @ f(x(t))
    Else it just returns zero.
    """
    try:
        jac = ivp.jacobian(t, x)
    except NotImplementedError:
        jac = np.zeros((len(x), len(x)))
    evl = ivp.rhs(t, x)
    if np.isscalar(evl) is True:
        evl = evl * np.ones(1)
    return jac @ evl
