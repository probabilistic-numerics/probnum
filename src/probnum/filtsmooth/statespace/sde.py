"""SDE models as transitions."""
import functools

import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv

from . import discrete_transition, transition


class SDE(transition.Transition):
    """Stochastic differential equation.

    .. math:: d x_t = g(t, x_t) d t + L(t) d w_t,

    driven by a Wiener process with unit diffusion.
    """

    def __init__(self, driftfun, dispmatrixfun, jacobfun):
        self._driftfun = driftfun
        self._dispmatrixfun = dispmatrixfun
        self._jacobfun = jacobfun

    def transition_realization(self, real, start, stop, **kwargs):
        raise NotImplementedError

    def transition_rv(self, rv, start, stop, **kwargs):
        raise NotImplementedError

    def drift(self, time, state, **kwargs):
        return self._driftfun(time, state, **kwargs)

    def dispersionmatrix(self, time, **kwargs):
        return self._dispmatrixfun(time, **kwargs)

    def jacobian(self, time, state, **kwargs):
        return self._jacobfun(time, state, **kwargs)

    @property
    def dimension(self):
        raise NotImplementedError


class LinearSDE(SDE):
    """Linear stochastic differential equation (SDE),

    .. math:: d x_t = [G(t) x_t + v(t)] d t + L(t) x_t d w_t.

    For Gaussian initial conditions, this solution is a Gaussian process.

    Parameters
    ----------
    driftmatrixfun : callable, signature=(t, \\**kwargs)
        This is F = F(t). The evaluations of this function are called
        the drift(matrix) of the SDE.
        Returns np.ndarray with shape=(n, n)
    forcevecfun : callable, signature=(t, \\**kwargs)
        This is u = u(t). Evaluations of this function are called
        the force(vector) of the SDE.
        Returns np.ndarray with shape=(n,)
    dispmatrixfun : callable, signature=(t, \\**kwargs)
        This is L = L(t). Evaluations of this function are called
        the dispersion(matrix) of the SDE.
        Returns np.ndarray with shape=(n, s)

    Notes
    -----
    If initial conditions are Gaussian, the solution is a Gauss-Markov process.
    """

    def __init__(self, driftmatrixfun, forcevecfun, dispmatrixfun):
        self._driftmatrixfun = driftmatrixfun
        self._forcevecfun = forcevecfun
        super().__init__(
            driftfun=(lambda t, x: driftmatrixfun(t) @ x + forcevecfun(t)),
            dispmatrixfun=dispmatrixfun,
            jacobfun=(lambda t, x: dispmatrixfun(t)),
        )

    def transition_realization(self, real, start, stop, step, **kwargs):
        rv = pnrv.Normal(real, 0 * np.eye(len(real)))
        return linear_sde_statistics(
            rv,
            start,
            stop,
            step,
            self._driftfun,
            self._driftmatrixfun,
            self._dispmatrixfun,
        )

    def transition_rv(self, rv, start, stop, step, **kwargs):

        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in linear SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        return linear_sde_statistics(
            rv,
            start,
            stop,
            step,
            self._driftfun,
            self._driftmatrixfun,
            self._dispmatrixfun,
        )

    @property
    def dimension(self):
        """Spatial dimension (utility attribute)."""
        return len(self._driftmatrixfun(0.0))


class LTISDE(LinearSDE):
    """Linear time-invariant continuous Markov models of the form
    dx = [F x(t) + u] dt + L dBt.
    In the language of dynamic models,
    x(t) : state process
    F : drift matrix
    u : forcing term
    L : dispersion matrix.
    Bt : Brownian motion with constant diffusion matrix Q.

    Parameters
    ----------
    driftmatrix : np.ndarray, shape=(n, n)
        This is F. It is the drift matrix of the SDE.
    forcevec : np.ndarray, shape=(n,)
        This is U. It is the force vector of the SDE.
    dispmatrix : np.ndarray, shape(n, s)
        This is L. It is the dispersion matrix of the SDE.

    Notes
    -----
    It assumes Gaussian initial conditions (otherwise
    it is no Gauss-Markov process).
    """

    def __init__(self, driftmatrix, forcevec, dispmatrix):
        _check_initial_state_dimensions(driftmatrix, forcevec, dispmatrix)
        super().__init__(
            (lambda t, **kwargs: driftmatrix),
            (lambda t, **kwargs: forcevec),
            (lambda t, **kwargs: dispmatrix),
        )
        self._driftmatrix = driftmatrix
        self._forcevec = forcevec
        self._dispmatrix = dispmatrix

    @property
    def driftmatrix(self):
        return self._driftmatrix

    @property
    def forcevec(self):
        return self._forcevec

    @property
    def dispersionmatrix(self):
        # pylint: disable=invalid-overridden-method
        return self._dispmatrix

    def transition_realization(self, real, start, stop, **kwargs):
        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        discretised_model = self.discretise(step=stop - start)
        return discretised_model.transition_realization(real, start, stop)

    def transition_rv(self, rv, start, stop, **kwargs):
        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        discretised_model = self.discretise(step=stop - start)
        return discretised_model.transition_rv(rv, start, stop)

    def discretise(self, step):
        """Returns a discrete transition model (i.e. mild solution to SDE) using matrix
        fraction decomposition.

        That is, matrices A(h) and Q(h) and vector s(h) such
        that the transition is

        .. math:: x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h)) ,

        which is the transition of the mild solution to the LTI SDE.
        """
        if np.linalg.norm(self._forcevec) > 0:
            raise NotImplementedError("MFD does not work for force>0 (yet).")
        ah, qh, _ = matrix_fraction_decomposition(
            self.driftmatrix, self.dispersionmatrix, step
        )
        sh = np.zeros(len(ah))
        return discrete_transition.DiscreteLTIGaussian(ah, sh, qh)


def _check_initial_state_dimensions(drift, force, disp):
    """Checks that the matrices all align and are of proper shape.

    If all the bugs are removed and the tests run, these asserts
    are turned into Exception-catchers.

    Parameters
    ----------
    drift : np.ndarray, shape=(n, n)
    force : np.ndarray, shape=(n,)
    disp : np.ndarray, shape=(n, s)
    """
    if drift.ndim != 2 or drift.shape[0] != drift.shape[1]:
        raise ValueError("driftmatrix not of shape (n, n)")
    if force.ndim != 1:
        raise ValueError("force not of shape (n,)")
    if force.shape[0] != drift.shape[1]:
        raise ValueError("force not of shape (n,) or driftmatrix not of shape (n, n)")
    if disp.ndim != 2:
        raise ValueError("dispersion not of shape (n, s)")


def linear_sde_statistics(rv, start, stop, step, driftfun, jacobfun, dispmatfun):
    """Computes mean and covariance of SDE solution.

    For a linear(ised) SDE

    .. math:: d x_t = [G(t) x_t + v(t)] d t + L(t) x_t d w_t.

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
    )
    rk4_step = functools.partial(_rk4_step, step=step, fun=increment_fun)

    while time < stop:
        mean, cov, time = rk4_step(mean, cov, time)
    return pnrv.Normal(mean, cov), {}


def _rk4_step(mean, cov, time, step, fun):
    """Do a single RK4 step to compute the solution."""
    m1, c1 = fun(time, mean, cov)
    m2, c2 = fun(time, mean + step * m1 / 2.0, cov + step * c1 / 2.0)
    m3, c3 = fun(time, mean + step * m2 / 2.0, cov + step * c2 / 2.0)
    m4, c4 = fun(time, mean + step * m3, cov + step * c3)
    mean = mean + step * (m1 + 2 * m2 + 2 * m3 + m4) / 6.0
    cov = cov + step * (c1 + 2 * c2 + 2 * c3 + c4) / 6.0
    time = time + step
    return mean, cov, time


def _increment_fun(time, mean, cov, driftfun, jacobfun, dispmatfun):
    """Euler step for closed form solutions of ODE defining mean and covariance of the
    closed-form transition.

    Maybe make this into a different solver (euler sucks).

    See RHS of Eq. 10.82 in Applied SDEs.
    """
    dispersion_matrix = dispmatfun(time)
    jacobian = jacobfun(time)
    mean_increment = driftfun(time, mean)
    cov_increment = (
        cov @ jacobian.T + jacobian @ cov.T + dispersion_matrix @ dispersion_matrix.T
    )
    return mean_increment, cov_increment


def matrix_fraction_decomposition(F, L, h):
    """Matrix fraction decomposition."""
    if F.ndim != 2 or L.ndim != 2:
        raise TypeError("F and L must be matrices.")
    if not np.isscalar(h):
        raise TypeError("h must be a float/scalar")

    topleft = F
    topright = L @ L.T
    bottomright = -F.T
    bottomleft = np.zeros(F.shape)

    toprow = np.hstack((topleft, topright))
    bottomrow = np.hstack((bottomleft, bottomright))
    bigmat = np.vstack((toprow, bottomrow))

    Phi = scipy.linalg.expm(bigmat * h)
    projmat1 = np.eye(*toprow.shape)
    projmat2 = np.flip(projmat1)

    Ah = projmat1 @ Phi @ projmat1.T
    C, D = projmat1 @ Phi @ projmat2.T, Ah.T
    Qh = C @ D

    return Ah, Qh, bigmat
