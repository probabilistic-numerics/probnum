"""
Adapter from initial value problems + state space model to
filters.
"""




import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.gaussfiltsmooth import extendedkalman
from probnum.filtsmooth.gaussfiltsmooth import unscentedkalman
from probnum.filtsmooth.statespace.discrete import DiscreteGaussianModel

__all__ = ["ivp_to_kf", "ivp_to_ekf", "ivp_to_ukf"]


def ivp_to_kf(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for KF based on IVP and prior.

    Returns ExtendedKalmanFilter object that is compatible with
    the GaussianIVPFilter.

    evlvar : float,
        measurement variance; in the literaturem, this is "R"
    """
    h0 = _h0(prior)
    measmod = _measmod_kf(ivp, h0, prior, evlvar)
    initdist = _initialdistribution(ivp, h0, prior)
    return extendedkalman.ExtendedKalmanFilter(prior, measmod,
                                    initdist)


def _measmod_kf(ivp, h0, prior, evlvar):
    """
    Zero-th order Taylor approximation as linearisation.

    We call it Kalman filter for convenience;
    it is no Kalman filter in reality.
    """
    ordint = prior.ordint
    spatialdim = prior.spatialdim
    h1_1d = np.eye(ordint + 1)[:, 1].reshape((1, ordint + 1))
    h1 = np.kron(np.eye(spatialdim), h1_1d)

    def dyna(t, x):
        return h1 @ x - ivp.rhs(t, h0 @ x)

    def diff(t):
        return evlvar * np.eye(spatialdim)

    def jaco(t, x):
        return h1

    return DiscreteGaussianModel(dyna, diff, jaco)


def ivp_to_ekf(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for EKF based on IVP and prior.

    Returns ExtendedKalmanFilter object.

    evlvar : float, (this is "R")
    """
    h0 = _h0(prior)
    measmod = _measmod_ekf(ivp, h0, prior, evlvar)
    initdist = _initialdistribution(ivp, h0, prior)
    return extendedkalman.ExtendedKalmanFilter(prior, measmod,
                                    initdist)


def _measmod_ekf(ivp, h0, prior, evlvar):
    """
    Computes H and R
    """
    ordint = prior.ordint
    spatialdim = prior.spatialdim
    h1_1d = np.eye(ordint + 1)[:, 1].reshape((1, ordint + 1))
    h1 = np.kron(np.eye(spatialdim), h1_1d)

    def dyna(t, x):
        return h1 @ x - ivp.rhs(t, h0 @ x)

    def diff(t):
        return evlvar * np.eye(spatialdim)

    def jaco(t, x):
        return h1 - ivp.jacobian(t, h0 @ x) @ h0

    return DiscreteGaussianModel(dyna, diff, jaco)


def ivp_to_ukf(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for EKF based on IVP and prior.

    Returns ExtendedKalmanFilter object.

    evlvar : float, (this is "R")
    """
    h0 = _h0(prior)
    measmod = _measmod_ukf(ivp, h0, prior, evlvar)
    initdist = _initialdistribution(ivp, h0, prior)
    return unscentedkalman.UnscentedKalmanFilter(prior, measmod,
                                    initdist, 1.0, 1.0, 1.0)


def _measmod_ukf(ivp, h0, prior, measvar):
    """
    """
    ordint = prior.ordint
    spatialdim = prior.spatialdim
    h1_1d = np.eye(ordint + 1)[:, 1].reshape((1, ordint + 1))
    h1 = np.kron(np.eye(spatialdim), h1_1d)

    def dyna(t, x):
        return h1 @ x - ivp.rhs(t, h0 @ x)

    def diff(t):
        return measvar * np.eye(spatialdim)

    return DiscreteGaussianModel(dyna, diff)


def _h0(prior):
    """
    Returns H0
    """
    ordint = prior.ordint
    spatialdim = prior.spatialdim
    h0_1d = np.eye(ordint + 1)[:, 0].reshape((1, ordint + 1))
    return np.kron(np.eye(spatialdim), h0_1d)


def _initialdistribution(ivp, h0, prior):
    """
    Eq. 39 in Schober et al.
    """
    initmean = _initialmean(ivp, h0, prior)
    initcovar = 0. * np.eye(len(initmean))
    return RandomVariable(distribution=Normal(initmean, initcovar))



def _initialmean(ivp, h0, prior):
    """
    """
    x0 = ivp.initialdistribution.mean()
    dx0 = ivp.rhs(0., x0)
    ddx0 = _ddx(0., x0, ivp)
    dddx0 = _dddx(0., x0, ivp)

    if prior.ordint == 1:  # (x0, f(x0))
        return _alternate2(x0, dx0)
    elif prior.ordint == 2:  # (x0, f(x0), ddx)
        return _alternate3(x0, dx0, ddx0)
    elif prior.ordint == 3:  # (x0, f(x0), ddx, dddx)
        return _alternate4(x0, dx0, ddx0, dddx0)
    else:
        raise NotImplementedError("Higher order methods require higher order derivatives of f")

def _ddx(t, x, ivp):
    """
    x''(t) = J_f(x(t)) @ f(x(t))
    """
    jac = ivp.jacobian(t, x)
    evl = ivp.rhs(t, x)
    return jac @ evl

def _dddx(t, x, ivp):
    """
    x'''(t) = f(x)^T @ H_f(x)^T @ f(x) + J_f(X)^T @ J_f(x) @ f(x)
    with an approximate Hessian-vector product.
    """
    rate = 1e-12
    jac = ivp.jacobian(t, x)
    evl = ivp.rhs(t, x)
    hess_at_f = (ivp.jacobian(0., x + rate*evl) - jac)
    return hess_at_f @ evl + jac.T @ jac @ evl


def _alternate2(arr1, arr2):
    """
    takes (a, b, c) and (d, e, f) into (a, d, b, e, c, f).
    """
    return np.vstack((arr1.T, arr2.T)).T.flatten()

def _alternate3(arr1, arr2, arr3):
    """
    takes (a, b, c) and (d, e, f) into (a, d, b, e, c, f).
    """
    return np.vstack((arr1.T, arr2.T, arr3.T)).T.flatten()


def _alternate4(arr1, arr2, arr3, arr4):
    """
    takes (a, b, c) and (d, e, f) into (a, d, b, e, c, f).
    """
    return np.vstack((arr1.T, arr2.T, arr3.T, arr4.T)).T.flatten()
