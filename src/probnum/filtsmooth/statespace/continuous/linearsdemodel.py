"""
Time continuous Gauss-Markov models implicitly defined
through being a solution to the SDE
dx(t) = F(t) x(t) dt + L(t) dB(t).

If initial condition is Gaussian RV, the solution
is a Gauss-Markov process.

Todo
----
chapmankolmogorov() is not tested!!!
"""

import numpy as np
import scipy.linalg

from probnum.quad import ClenshawCurtis
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace.continuous import continuousmodel


__all__ = ["LinearSDEModel", "LTISDEModel"]


class LinearSDEModel(continuousmodel.ContinuousModel):
    """
    Linear time-continuous Markov models of the form
    dx = [F(t) x(t) + u(t)] dt + L(t) dB(t).
    In the language of dynamic models,
    x(t) : state process
    F(t) : drift matrix
    u(t) : forcing term
    L(t) : dispersion matrix.
    B(t) : Brownian motion with diffusion matrix Q.

    Note
    ----
    If initial conditions are Gaussian, it is a
    Gauss-Markov process.
    We assume Gaussianity for chapmankolmogorov()
    """

    def __init__(self, driftmatrixfct, forcfct, dispmatrixfct, diffmatrix):
        """
        Arguments
        ---------
        driftmatrixfct : callable, signature (t, *args, **kwargs)
            maps t -> F(t)
        forcfct : callable, signature (t, *args, **kwargs)
            maps t -> u(t)
        dispmatrixfct : callable, signature (t, *args, **kwargs)
            maps t -> L(t)
        diffmatrix : np.ndarray, shape (d, d)
            Diffusion matrix Q
        """
        self._driftmatrixfct = driftmatrixfct
        self._forcefct = forcfct
        self._dispmatrixfct = dispmatrixfct
        self._diffmatrix = diffmatrix

    def drift(self, time, state, *args, **kwargs):
        """
        Evaluates f(t, x(t)) = F(t) x(t) + u(t).
        """
        driftmatrix = self._driftmatrixfct(time, *args, **kwargs)
        force = self._forcefct(time, *args, **kwargs)
        return driftmatrix @ state + force

    def dispersion(self, time, state, *args, **kwargs):
        """
        Evaluates l(t, x(t)) = L(t).
        """
        return self._dispmatrixfct(time, *args, **kwargs)

    def jacobian(self, time, state, *args, **kwargs):
        """
        maps t -> F(t)
        """
        return self._driftmatrixfct(time, *args, **kwargs)

    @property
    def diffusionmatrix(self):
        """
        Evaluates Q.
        """
        return self._diffmatrix

    def chapmankolmogorov(self, start, stop, step, randvar, *args, **kwargs):
        """
        Solves differential equations for mean and
        covariance of the SDE solution (Eq. 5.50 and 5.51
        or Eq. 10.73 in Applied SDEs).

        By default, we assume that randvar is Gaussian.
        """
        mean, covar = randvar.mean(), randvar.cov()
        time = start
        while time < stop:
            meanincr, covarincr = self._iterate(time, mean, covar, *args,
                                                **kwargs)
            mean, covar = mean + step * meanincr, covar + step * covarincr
            time = time + step
        return RandomVariable(distribution=Normal(mean, covar))

    def _iterate(self, time, mean, covar, *args, **kwargs):
        """
        RHS of Eq. 10.82 in Applied SDEs
        """
        disped = self.dispersion(time, mean, *args, **kwargs)
        jacob = self.jacobian(time, mean, *args, **kwargs)
        diff = self.diffusionmatrix
        newmean = self.drift(time, mean, *args, **kwargs)
        newcovar = covar @ jacob.T + jacob @ covar.T + disped @ diff @ disped.T
        return newmean, newcovar


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

    Note
    ----
    It assumes Gaussian initial conditions (otherwise
    it is no Gauss-Markov process).
    """

    def __init__(self, driftmatrix, force, dispmatrix, diffmatrix):
        """
        Arguments
        ---------
        driftmatrix : ndarray (F)
        force : ndarray (u)
        dispmatrix : ndarray (L)
        diffmatrix : ndarray (Q)
        """
        super().__init__((lambda t, *args, **kwargs: driftmatrix),
                                (lambda t, *args, **kwargs: force),
                                (lambda t, *args, **kwargs: dispmatrix),
                                diffmatrix)
        self._driftmatrix = driftmatrix
        self._force = force
        self._dispmatrix = dispmatrix
        self._diffmatrix = diffmatrix

    @property
    def driftmatrix(self):
        """
        """
        return self._driftmatrix

    @property
    def force(self):
        """
        """
        return self._force

    @property
    def dispersionmatrix(self):
        """
        """
        return self._dispmatrix

    def chapmankolmogorov(self, start, stop, step, randvar, *args, **kwargs):
        """
        Overwrites StateSpaceComponent.chapmankolmogorov()
        since for linear SDEs, there exists a closed form
        solution.

        Closed form solution for mean and
        covariance of the SDE solution, Eq. 6.9 to Eq. 6.11
        in Applied SDEs.
        """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        h = stop - start
        nsteps = int((h) / step)
        if nsteps % 2 == 0:
            nsteps = nsteps + 1
        quad = ClenshawCurtis(nsteps, 1, np.array([[0, h]]))
        drift, disp, diff = self.driftmatrix, self.dispersionmatrix, self.diffusionmatrix

        def integ1(x):
            return scipy.linalg.expm(drift * (h - x))

        def integ2(x):
            return np.outer(integ1(x) @ disp,
                                       diff @ integ1(x) @ disp)

        trans = integ1(0)
        force = quad.integrate(lambda x: integ1(x) @ self.force, isvectorized=False)
        transdiff = quad.integrate(integ2, isvectorized=False)
        newmean = trans @ mean + force
        newcov = trans @ covar @ trans.T + transdiff
        return RandomVariable(distribution=Normal(newmean, newcov))

