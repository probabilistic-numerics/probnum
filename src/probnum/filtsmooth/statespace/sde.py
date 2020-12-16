"""SDE models as transitions."""
import functools
from typing import Callable

import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import discrete_transition, transition


class SDE(transition.Transition):
    """Stochastic differential equation.

    .. math:: d x(t) = g(t, x(t)) d t + L(t) d w(t),

    driven by a Wiener process with unit diffusion.
    """

    def __init__(
        self,
        driftfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        dispmatfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        jacobfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
    ):
        self.driftfun = driftfun
        self.dispmatfun = dispmatfun
        self.jacobfun = jacobfun
        super().__init__()

    def transition_realization(
        self,
        real,
        start,
        stop=None,
        step=None,
        linearise_at=None,
    ):
        raise NotImplementedError

    def transition_rv(
        self,
        rv,
        start,
        stop=None,
        step=None,
        linearise_at=None,
    ):
        raise NotImplementedError

    @property
    def dimension(self):
        raise NotImplementedError


class LinearSDE(SDE):
    """Linear stochastic differential equation (SDE),

    .. math:: d x(t) = [G(t) x(t) + v(t)] d t + L(t) x(t) d w(t).

    For Gaussian initial conditions, this solution is a Gaussian process.

    Parameters
    ----------
    driftmatfun :
        This is G = G(t). The evaluations of this function are called
        the driftmatrix of the SDE.
        Returns np.ndarray with shape=(n, n)
    forcevecfun :
        This is v = v(t). Evaluations of this function are called
        the force(vector) of the SDE.
        Returns np.ndarray with shape=(n,)
    dispmatfun :
        This is L = L(t). Evaluations of this function are called
        the dispersion(matrix) of the SDE.
        Returns np.ndarray with shape=(n, s)
    """

    def __init__(
        self,
        driftmatfun: Callable[[FloatArgType], np.ndarray],
        forcevecfun: Callable[[FloatArgType], np.ndarray],
        dispmatfun: Callable[[FloatArgType], np.ndarray],
    ):
        self.driftmatfun = driftmatfun
        self.forcevecfun = forcevecfun
        super().__init__(
            driftfun=(lambda t, x: driftmatfun(t) @ x + forcevecfun(t)),
            dispmatfun=dispmatfun,
            jacobfun=(lambda t, x: driftmatfun(t)),
        )

    def transition_realization(
        self,
        real,
        start,
        stop,
        step,
        **kwargs,
    ):

        rv = pnrv.Normal(real, 0 * np.eye(len(real)))
        return linear_sde_statistics(
            rv,
            start,
            stop,
            step,
            self.driftfun,
            self.driftmatfun,
            self.dispmatfun,
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
            self.driftfun,
            self.driftmatfun,
            self.dispmatfun,
        )

    @property
    def dimension(self):
        """Spatial dimension (utility attribute)."""
        # risky to evaluate at zero, but usually works
        return len(self.driftmatfun(0.0))


class LTISDE(LinearSDE):
    """Linear time-invariant continuous Markov models of the form.

    .. math:: d x(t) = [G x(t) + v] d t + L d w(t).

    In the language of dynamic models,
    x(t) : state process
    G : drift matrix
    v : force term/vector
    L : dispersion matrix.
    w(t) : Wiener process with unit diffusion.

    Parameters
    ----------
    driftmat :
        This is F. It is the drift matrix of the SDE.
    forcevec :
        This is U. It is the force vector of the SDE.
    dispmat :
        This is L. It is the dispersion matrix of the SDE.
    """

    def __init__(self, driftmat: np.ndarray, forcevec: np.ndarray, dispmat: np.ndarray):
        _check_initial_state_dimensions(driftmat, forcevec, dispmat)
        super().__init__(
            (lambda t: driftmat),
            (lambda t: forcevec),
            (lambda t: dispmat),
        )
        self.driftmat = driftmat
        self.forcevec = forcevec
        self.dispmat = dispmat

    def transition_realization(
        self,
        real,
        start,
        stop,
        **kwargs,
    ):

        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        discretised_model = self.discretise(step=stop - start)
        return discretised_model.transition_realization(real, start)

    def transition_rv(
        self,
        rv,
        start,
        stop,
        **kwargs,
    ):

        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        discretised_model = self.discretise(step=stop - start)
        return discretised_model.transition_rv(rv, start)

    def discretise(self, step):
        """Returns a discrete transition model (i.e. mild solution to SDE) using matrix
        fraction decomposition.

        That is, matrices A(h) and Q(h) and vector s(h) such
        that the transition is

        .. math:: x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h)) ,

        which is the transition of the mild solution to the LTI SDE.
        """
        if np.linalg.norm(self.forcevec) > 0:
            raise NotImplementedError("MFD does not work for force>0 (yet).")
        ah, qh, _ = matrix_fraction_decomposition(self.driftmat, self.dispmat, step)
        sh = np.zeros(len(ah))
        return discrete_transition.DiscreteLTIGaussian(ah, sh, qh)


def linear_sde_statistics(rv, start, stop, step, driftfun, jacobfun, dispmatfun):
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


def _check_initial_state_dimensions(driftmat, forcevec, dispmat):
    """Checks that the matrices all align and are of proper shape.

    Parameters
    ----------
    driftmat : np.ndarray, shape=(n, n)
    forcevec : np.ndarray, shape=(n,)
    dispmat : np.ndarray, shape=(n, s)
    """
    if driftmat.ndim != 2 or driftmat.shape[0] != driftmat.shape[1]:
        raise ValueError("driftmatrix not of shape (n, n)")
    if forcevec.ndim != 1:
        raise ValueError("force not of shape (n,)")
    if forcevec.shape[0] != driftmat.shape[1]:
        raise ValueError("force not of shape (n,) or driftmatrix not of shape (n, n)")
    if dispmat.ndim != 2:
        raise ValueError("dispersion not of shape (n, s)")
