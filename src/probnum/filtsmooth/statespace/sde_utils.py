"""Continuous-time transition implementations.

All sorts of implementations for solving moment equations.
"""


def linear_sde_statistics(
    rv,
    start,
    stop,
    step,
    driftfun,
    jacobfun,
    dispmatfun,
    _diffusion: Optional[float] = 1.0,
):
    """Computes mean and covariance of SDE solution.

    For a linear(ised) SDE

    .. math:: d x(t) = [G(t) x(t) + v(t)] d t + L(t) x(t) d w(t).

    mean and covariance of the solution are computed by solving

    .. math:: \\frac{dm}{dt}(t) = G(t) m(t) + v(t), \\frac{dC}{dt}(t) = G(t) C(t) + C(t) G(t)^\\top + L(t) L(t)^\\top,

    which is done here with a few steps of the RK4 method.
    This function is also called by the continuous-time extended Kalman filter,
    which is why the drift can be any function.

    Parameters
    ----------
    rv :
        Normal random variable. Distribution of mean and covariance at the starting point.
    start :
        Start of the time-interval
    stop :
        End of the time-interval
    step :
        Step-size used in RK4.
    driftfun :
        Drift of the (non)linear SDE
    jacobfun :
        Jacobian of the drift function
    dispmatfun :
        Dispersion matrix function

    Returns
    -------
    Normal random variable
        Mean and covariance are the solution of the differential equation
    dict
        Empty dict, may be extended in the future to contain information
        about the solution process, e.g. number of function evaluations.
    """
    if step <= 0.0:
        raise ValueError("Step-size must be positive.")
    mean, cov = rv.mean, rv.cov
    time = start

    # Set default arguments for frequently used functions.
    increment_fun = functools.partial(
        _increment_fun,
        driftfun=driftfun,
        jacobfun=jacobfun,
        dispmatfun=dispmatfun,
        _diffusion=_diffusion,
    )
    rk4_step = functools.partial(_rk4_step, step=step, fun=increment_fun)

    while time < stop:
        mean, cov, time = rk4_step(mean, cov, time)
    return pnrv.Normal(mean, cov), {}


def _rk4_step(mean, cov, time, step, fun):
    """Do a single RK4 step to compute the solution."""
    m1, c1 = fun(time, mean, cov)
    m2, c2 = fun(time + step / 2.0, mean + step * m1 / 2.0, cov + step * c1 / 2.0)
    m3, c3 = fun(time + step / 2.0, mean + step * m2 / 2.0, cov + step * c2 / 2.0)
    m4, c4 = fun(time + step, mean + step * m3, cov + step * c3)
    mean = mean + step * (m1 + 2 * m2 + 2 * m3 + m4) / 6.0
    cov = cov + step * (c1 + 2 * c2 + 2 * c3 + c4) / 6.0
    time = time + step
    return mean, cov, time


def _increment_fun(
    time, mean, cov, driftfun, jacobfun, dispmatfun, _diffusion: Optional[float] = 1.0
):
    """Euler step for closed form solutions of ODE defining mean and covariance of the
    closed-form transition.

    Maybe make this into a different solver (euler sucks).

    See RHS of Eq. 10.82 in Applied SDEs.
    """
    dispersion_matrix = dispmatfun(time)
    jacobian = jacobfun(time)
    mean_increment = driftfun(time, mean)
    cov_increment = (
        cov @ jacobian.T
        + jacobian @ cov.T
        + _diffusion * dispersion_matrix @ dispersion_matrix.T
    )
    return mean_increment, cov_increment


def matrix_fraction_decomposition(driftmat, dispmat, step):
    """Matrix fraction decomposition (without force)."""
    no_force = np.zeros(len(driftmat))
    _check_initial_state_dimensions(
        driftmat=driftmat, forcevec=no_force, dispmat=dispmat
    )

    topleft = driftmat
    topright = dispmat @ dispmat.T
    bottomright = -driftmat.T
    bottomleft = np.zeros(driftmat.shape)

    toprow = np.hstack((topleft, topright))
    bottomrow = np.hstack((bottomleft, bottomright))
    bigmat = np.vstack((toprow, bottomrow))

    Phi = scipy.linalg.expm(bigmat * step)
    projmat1 = np.eye(*toprow.shape)
    projmat2 = np.flip(projmat1)

    Ah = projmat1 @ Phi @ projmat1.T
    C, D = projmat1 @ Phi @ projmat2.T, Ah.T
    Qh = C @ D

    return Ah, Qh, bigmat
