"""
Adapter methods:
from initial value problems + state space model to filters.
"""

import numpy as np

from probnum.filtsmooth import ExtendedKalman, UnscentedKalman
from probnum.filtsmooth.statespace.discrete import DiscreteGaussianModel
from probnum import random_variables as rvs


def ivp2ekf0(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for KF based on IVP and prior.

    **Initialdistribution:**

    Conditions the initial distribution of the Gaussian filter
    onto the initial values.

    - If preconditioning is set to ``False``, it conditions
      the initial distribution :math:`\\mathcal{N}(0, I)`
      on the initial values :math:`(x_0, f(t_0, x_0), ...)` using
      as many available deri    vatives as possible.

    - If preconditioning is set to ``True``, it conditions
      the initial distribution :math:`\\mathcal{N}(0, P P^\\top)`
      on the initial values :math:`(x_0, f(t_0, x_0), ...)` using
      as many available derivatives as possible.
      Note that the projection matrices :math:`H_0` and :math:`H_1`
      become :math:`H_0 P^{-1}` and :math:`H_1 P^{-1}` which has
      to be taken into account during the preconditioning.

    **Measurement model:**

    Returns a measurement model :math:`\\mathcal{N}(g(m), R)`
    involving computing the discrepancy

    .. math:: g(m) = H_1 m(t) - f(t, H_0 m(t)).

    Then it returns either type of Gaussian filter, each with a
    different interpretation of the Jacobian :math:`J_g`:

    - EKF0 thinks :math:`J_g(m) = H_1`
    - EKF1 thinks :math:`J_g(m) = H_1 - J_f(t, H_0 m(t)) H_0^\\top`
    - UKF thinks: ''What is a Jacobian?''

    Note that, again, in the case of a preconditioned state space
    model, :math:`H_0` and :math:`H_1`
    become :math:`H_0 P^{-1}` and :math:`H_1 P^{-1}` which has
    to be taken into account. In this case,

    - EKF0 thinks :math:`J_g(m) = H_1 P^{-1}`
    - EKF1 thinks :math:`J_g(m) = H_1 P^{-1} - J_f(t, H_0  P^{-1} m(t)) (H_0 P^{-1})^\\top`
    - UKF again thinks: ''What is a Jacobian?''

    **Note:**
    The choice between :math:`H_i` and :math:`H_i P^{-1}` is taken care
    of within the Prior.

    Returns ExtendedKalmanFilter object that is compatible with
    the GaussianIVPFilter.

    evlvar : float,
        measurement variance; in the literature, this is "R"
    """  # pylint: disable=line-too-long
    measmod = _measmod_ekf0(ivp, prior, evlvar)
    initrv = _initialdistribution(ivp, prior)
    return ExtendedKalman(prior, measmod, initrv)


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


def ivp2ekf1(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for EKF based on IVP and prior.

    Returns ExtendedKalmanFilter object.

    evlvar : float, (this is "R")
    """
    measmod = _measmod_ekf1(ivp, prior, evlvar)
    initrv = _initialdistribution(ivp, prior)
    return ExtendedKalman(prior, measmod, initrv)


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


def ivp2ukf(ivp, prior, evlvar):
    """
    Computes measurement model and initial distribution
    for EKF based on IVP and prior.

    Returns ExtendedKalmanFilter object.

    evlvar : float, (this is "R")
    """
    measmod = _measmod_ukf(ivp, prior, evlvar)
    initrv = _initialdistribution(ivp, prior)
    return UnscentedKalman(prior, measmod, initrv, 1.0, 1.0, 1.0)


def _measmod_ukf(ivp, prior, measvar):

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
    Conditions initialdistribution :math:`\\mathcal{N}(0, P P^\\top)`
    on the initial values :math:`(x_0, f(t_0, x_0), ...)` using
    as many available derivatives as possible.

    Note that the projection matrices :math:`H_0` and :math:`H_1`
    become :math:`H_0 P^{-1}` and :math:`H_1 P^{-1}`.
    """
    if not issubclass(type(ivp.initrv), rvs.Normal):
        if not issubclass(type(ivp.initrv), rvs.Dirac):
            raise RuntimeError("Initial distribution not Normal nor Dirac")
    x0 = ivp.initialdistribution.mean
    dx0 = ivp.rhs(ivp.t0, x0)
    ddx0 = _ddx(ivp.t0, x0, ivp)
    dddx0 = _dddx(ivp.t0, x0, ivp)
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)
    precond = prior.preconditioner
    # initcov = np.eye(*(precond @ precond.T).shape)
    initcov = precond @ precond.T
    if prior.ordint == 1:
        projmat = np.hstack((h0.T, h1.T)).T
        data = np.hstack((x0, dx0))
        _size = 2
    elif prior.ordint == 2:  # try only jacobian
        h2 = prior.proj2coord(coord=2)
        projmat = np.hstack((h0.T, h1.T, h2.T)).T
        data = np.hstack((x0, dx0, ddx0))
        _size = 3
    else:  # try jacobian and hessian
        h2 = prior.proj2coord(coord=2)
        h3 = prior.proj2coord(coord=3)
        projmat = np.hstack((h0.T, h1.T, h2.T, h3.T)).T
        data = np.hstack((x0, dx0, ddx0, dddx0))
        _size = 4
    largecov = np.kron(np.eye(_size), ivp.initialdistribution.cov)
    s = projmat @ initcov @ projmat.T + largecov
    crosscov = initcov @ projmat.T
    newmean = crosscov @ np.linalg.solve(s, data)
    newcov = initcov - (crosscov @ np.linalg.solve(s.T, crosscov.T)).T
    return rvs.Normal(newmean, newcov)


def _initialdistribution_no_precond(ivp, prior):
    x0 = ivp.initialdistribution.mean
    dx0 = ivp.rhs(ivp.t0, x0)
    ddx0 = _ddx(ivp.t0, x0, ivp)
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)
    initcov = np.eye(len(h0.T))
    if prior.ordint == 1:
        projmat = np.hstack((h0.T, h1.T)).T
        data = np.hstack((x0, dx0))
    else:
        h2 = prior.proj2coord(coord=2)
        projmat = np.hstack((h0.T, h1.T, h2.T)).T
        data = np.hstack((x0, dx0, ddx0))
    s = projmat @ initcov @ projmat.T
    crosscov = initcov @ projmat.T  # @ np.linalg.inv(s)
    newmean = crosscov @ np.linalg.solve(s, data)
    newcov = initcov - (crosscov @ np.linalg.solve(s, crosscov)).T
    return rvs.Normal(newmean, newcov)


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
        evl = np.array([evl])
        jac = np.array([jac])
    return jac @ evl


def _dddx(t, x, ivp):
    """
    x'''(t) = H_f(x) @ f(x) @ f(x) + J_f(X) @ J_f(x) @ f(x)
    with an approximate Hessian-vector product.
    """
    evl = ivp.rhs(t, x)
    try:
        jac = ivp.jacobian(t, x)
    except NotImplementedError:
        jac = np.zeros((len(x), len(x)))
    try:
        hess = ivp.hessian(t, x)
    except NotImplementedError:
        hess = np.zeros((len(x), len(x), len(x)))
    if np.isscalar(evl):
        evl = np.array([evl])
        jac = np.array([jac])
        hess = np.array([hess])
    return (hess @ evl) @ evl + jac @ (jac @ evl)
