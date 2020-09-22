"""
Time continuous Gauss-Markov models implicitly defined
through being a solution to the SDE
dx(t) = F(t) x(t) dt + L(t) dB(t).

If initial condition is Gaussian RV, the solution
is a Gauss-Markov process.
"""

import numpy as np
import scipy.linalg
from probnum.filtsmooth.statespace.continuous import continuousmodel
from probnum.random_variables import Normal

__all__ = ["LinearSDEModel", "LTISDEModel"]


class LinearSDEModel(continuousmodel.ContinuousModel):
    """
    Linear time-continuous Markov models given by the solution of the
    stochastic differential equation
    :math:`dx = [F(t) x(t) + u(t)] dt + L(t) dB(t)`.


    Parameters
    ----------
    driftmatrixfct : callable, signature=(t, \\**kwargs)
        This is F = F(t). The evaluations of this function are called
        the drift(matrix) of the SDE.
        Returns np.ndarray with shape=(n, n)
    forcfct : callable, signature=(t, \\**kwargs)
        This is u = u(t). Evaluations of this function are called
        the force(vector) of the SDE.
        Returns np.ndarray with shape=(n,)
    dispmatrixfct : callable, signature=(t, \\**kwargs)
        This is L = L(t). Evaluations of this function are called
        the dispersion(matrix) of the SDE.
        Returns np.ndarray with shape=(n, s)
    diffmatrix : np.ndarray, shape=(s, s)
        This is the diffusion matrix Q of the Brownian motion.
        It is always a square matrix and the size of this matrix matches
        the number of columns of the dispersionmatrix.

    Notes
    -----
    If initial conditions are Gaussian, the solution is a Gauss-Markov process.
    We assume Gaussianity for :meth:`chapmankolmogorov`.
    """

    def __init__(self, driftmatrixfct, forcfct, dispmatrixfct, diffmatrix):
        self._driftmatrixfct = driftmatrixfct
        self._forcefct = forcfct
        self._dispmatrixfct = dispmatrixfct
        self._diffmatrix = diffmatrix

    def transition_realization(self, real, start, stop, **kwargs):
        step = kwargs["step"]
        rv = Normal(real, 0 * np.eye(len(real)))
        return self._solve_chapmankolmogorov_equations(
            start=start, stop=stop, step=step, randvar=rv
        )

    def transition_rv(self, rv, start, stop, **kwargs):
        step = kwargs["step"]
        if not issubclass(type(rv), Normal):
            errormsg = (
                "Closed form solution for Chapman-Kolmogorov "
                "equations in linear SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise ValueError(errormsg)
        return self._solve_chapmankolmogorov_equations(
            start=start, stop=stop, step=step, randvar=rv
        )

    def _solve_chapmankolmogorov_equations(self, start, stop, step, randvar, **kwargs):
        """
        Solves differential equations for mean and
        kernels of the SDE solution (Eq. 5.50 and 5.51
        or Eq. 10.73 in Applied SDEs).

        By default, we assume that ``randvar`` is Gaussian.
        """
        mean, covar = randvar.mean, randvar.cov
        time = start
        while time < stop:
            meanincr, covarincr = self._increment(time, mean, covar, **kwargs)
            mean, covar = mean + step * meanincr, covar + step * covarincr
            time = time + step
        return Normal(mean, covar), {}

    def _increment(self, time, mean, covar, **kwargs):
        """
        Euler step for closed form solutions of ODE defining mean
        and kernels of the solution of the Chapman-Kolmogoro
        equations (via Fokker-Planck equations, but that is not crucial
        here).
        See RHS of Eq. 10.82 in Applied SDEs.
        """
        disped = self.dispersion(time, mean, **kwargs)
        jacob = self.jacobian(time, mean, **kwargs)
        diff = self.diffusionmatrix
        newmean = self.drift(time, mean, **kwargs)
        newcovar = covar @ jacob.T + jacob @ covar.T + disped @ diff @ disped.T
        return newmean, newcovar

    def drift(self, time, state, **kwargs):
        """
        Evaluates f(t, x(t)) = F(t) x(t) + u(t).
        """
        driftmatrix = self._driftmatrixfct(time, **kwargs)
        force = self._forcefct(time, **kwargs)
        return driftmatrix @ state + force

    def dispersion(self, time, state, **kwargs):
        """
        Evaluates l(t, x(t)) = L(t).
        """
        return self._dispmatrixfct(time, **kwargs)

    def jacobian(self, time, state, **kwargs):
        """
        maps t -> F(t)
        """
        return self._driftmatrixfct(time, **kwargs)

    @property
    def diffusionmatrix(self):
        """
        Evaluates Q.
        """
        return self._diffmatrix

    @property
    def dimension(self):
        """
        Spatial dimension (utility attribute).
        """
        return len(self._driftmatrixfct(0.0))


class LTISDEModel(LinearSDEModel):
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
    force : np.ndarray, shape=(n,)
        This is U. It is the force vector of the SDE.
    dispmatrix : np.ndarray, shape(n, s)
        This is L. It is the dispersion matrix of the SDE.
    diffmatrix : np.ndarray, shape=(s, s)
        This is the diffusion matrix Q of the Brownian motion
        driving the SDE.

    Notes
    -----
    It assumes Gaussian initial conditions (otherwise
    it is no Gauss-Markov process).
    """

    def __init__(self, driftmatrix, force, dispmatrix, diffmatrix):
        """
        Parameters
        ----------
        driftmatrix : ndarray (F)
        force : ndarray (u)
        dispmatrix : ndarray (L)
        diffmatrix : ndarray (Q)
        """
        _check_initial_state_dimensions(driftmatrix, force, dispmatrix, diffmatrix)
        super().__init__(
            (lambda t, **kwargs: driftmatrix),
            (lambda t, **kwargs: force),
            (lambda t, **kwargs: dispmatrix),
            diffmatrix,
        )
        self._driftmatrix = driftmatrix
        self._force = force
        self._dispmatrix = dispmatrix
        self._diffmatrix = diffmatrix

    @property
    def driftmatrix(self):
        return self._driftmatrix

    @property
    def force(self):
        return self._force

    @property
    def dispersionmatrix(self):
        return self._dispmatrix

    def transition_realization(self, real, start, stop, **kwargs):
        if not isinstance(real, np.ndarray):
            raise TypeError
        disc_dynamics, disc_force, disc_diffusion = self._discretise(
            step=(stop - start)
        )
        return Normal(disc_dynamics @ real + disc_force, disc_diffusion), {}

    def transition_rv(self, rv, start, stop, **kwargs):
        if not isinstance(rv, Normal):
            errormsg = (
                "Closed form solution for Chapman-Kolmogorov "
                "equations in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        disc_dynamics, disc_force, disc_diffusion = self._discretise(
            step=(stop - start)
        )
        old_mean, old_cov = rv.mean, rv.cov
        new_mean = disc_dynamics @ old_mean + disc_force
        new_crosscov =  old_cov @ disc_dynamics.T
        new_cov = disc_dynamics @ new_crosscov + disc_diffusion
        return Normal(mean=new_mean, cov=new_cov), {"crosscov": new_crosscov}

    def _discretise(self, step):
        """
        Returns discretised model (i.e. mild solution to SDE)
        using matrix fraction decomposition. That is, matrices A(h)
        and Q(h) and vector s(h) such that the transition is

        .. math::`x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h))`

        which is the transition of the mild solution to the LTI SDE.
        """
        blockmat, proj = self._form_driftmatrix_extended_state()
        expm = scipy.linalg.expm(step * blockmat)
        ah = proj @ expm @ proj.T
        sh = proj @ expm @ np.flip(proj).T @ self.force
        qh = proj @ expm @ np.flip(proj).T @ ah.T
        return ah, sh, qh

    def _form_driftmatrix_extended_state(self):
        """
        Forms the driftmatrix for state space model (x, u),
        i.e. F = (F, I; 0; 0).

        Returns blockmatrix and projection to $F$
        """
        drift = self.driftmatrix
        disp = self.dispersionmatrix
        diff = self.diffusionmatrix
        firstrowblock = np.hstack((drift, disp @ diff @ disp.T))
        secondrowblock = np.hstack((0 * drift.T, -1.0 * drift.T))
        blockmat = np.hstack((firstrowblock.T, secondrowblock.T)).T
        proj = np.eye(*firstrowblock.shape)
        return blockmat, proj


def _check_initial_state_dimensions(drift, force, disp, diff):
    """
    Checks that the matrices all align and are of proper shape.

    If all the bugs are removed and the tests run, these asserts
    are turned into Exception-catchers.

    Parameters
    ----------
    drift : np.ndarray, shape=(n, n)
    force : np.ndarray, shape=(n,)
    disp : np.ndarray, shape=(n, s)
    diff : np.ndarray, shape=(s, s)

    """
    if drift.ndim != 2 or drift.shape[0] != drift.shape[1]:
        raise ValueError("driftmatrix not of shape (n, n)")
    if force.ndim != 1:
        raise ValueError("force not of shape (n,)")
    if force.shape[0] != drift.shape[1]:
        raise ValueError("force not of shape (n,)" "or driftmatrix not of shape (n, n)")
    if disp.ndim != 2:
        raise ValueError("dispersion not of shape (n, s)")
    if diff.ndim != 2 or diff.shape[0] != diff.shape[1]:
        raise ValueError("diffusion not of shape (s, s)")
    if disp.shape[1] != diff.shape[0]:
        raise ValueError(
            "dispersion not of shape (n, s)" "or diffusion not of shape (s, s)"
        )
