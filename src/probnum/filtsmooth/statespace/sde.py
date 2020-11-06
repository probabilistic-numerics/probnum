"""SDE models as transitions."""
import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv
from probnum.filtsmooth.statespace import discrete_transition, transition


class SDE(transition.Transition):
    """
    Stochastic differential equation.

    .. math:: d x_t = g(t, x_t) d t + l(t, x_t) d w_t,

    driven by a Wiener process with unit diffusion.
    """

    def __init__(self, driftfun, dispersionfun, jacobfun):
        self._driftfun = driftfun
        self._dispersionfun = dispersionfun
        self._jacobfun = jacobfun

    def transition_realization(self, real, start, stop, **kwargs):
        raise NotImplementedError

    def transition_rv(self, rv, start, stop, **kwargs):
        raise NotImplementedError

    def drift(self, time, state, **kwargs):
        return self._driftfun(time, state, **kwargs)

    def dispersion(self, time, state, **kwargs):
        return self._dispersionfun(time, state, **kwargs)

    def jacobian(self, time, state, **kwargs):
        return self._jacobfun(time, state, **kwargs)

    @property
    def dimension(self):
        raise NotImplementedError


class LinearSDE(SDE):
    """
    Linear stochastic differential equation (SDE),

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
        self._dispmatrixfun = dispmatrixfun
        super().__init__(
            driftfun=(lambda t, x: driftmatrixfun(t) @ x + forcevecfun(t)),
            dispersionfun=(lambda t, x: dispmatrixfun(t) @ x),
            jacobfun=(lambda t, x: dispmatrixfun(t)),
        )

    def transition_realization(self, real, start, stop, euler_step, **kwargs):
        rv = pnrv.Normal(real, 0 * np.eye(len(real)))
        return linear_sde_statistics(
            rv,
            start,
            stop,
            euler_step,
            self._driftfun,
            self._driftmatrixfun,
            self._dispmatrixfun,
        )

    def transition_rv(self, rv, start, stop, euler_step, **kwargs):

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
            euler_step,
            self._driftfun,
            self._driftmatrixfun,
            self._dispmatrixfun,
        )

    @property
    def dimension(self):
        """
        Spatial dimension (utility attribute).
        """
        return len(self._driftmatrixfun(0.0))


class LTISDE(LinearSDE):
    """
    Linear time-invariant continuous Markov models of the
    form
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
        """
        Returns a discrete transition model (i.e. mild solution to SDE)
        using matrix fraction decomposition.

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
    """
    Checks that the matrices all align and are of proper shape.

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
    """Computes mean and covariance of SDE solution."""
    if step <= 0.0:
        raise ValueError("Step-size must be positive.")
    mean, cov = rv.mean, rv.cov
    time = start
    while time < stop:
        meanincr, covincr = _evaluate_increments(
            time, mean, cov, driftfun, jacobfun, dispmatfun
        )
        mean, cov = mean + step * meanincr, cov + step * covincr
        time = time + step
    return pnrv.Normal(mean, cov), {}


def _evaluate_increments(time, mean, cov, driftfun, jacobfun, dispmatfun):
    """
    Euler step for closed form solutions of ODE defining mean
    and covariance of the closed-form transition.

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
