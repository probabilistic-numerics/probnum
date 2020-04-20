"""
Continuous-Time priors for ODE solvers.

Currently, they are only relevant in the context of ODEs.
If needed in a more general setting, it is easy to move
them to statespace module (->thoughts?)

Matern will be easy to implement, just reuse the template
provided by IOUP and change parameters
"""
import numpy as np
from scipy.special import binom   # for Matern

from probnum.filtsmooth.statespace.continuous import LTISDEModel
from probnum.prob import RandomVariable, Normal

class ODEPrior(LTISDEModel):
    """
    Prior dynamic model for ODE filtering and smoothing.

    An ODE prior is an LTI state space model with specific attributes:
        * order of integration
        * spatial dimension of the underlying ODE
        * projection to X_0 (the state estimate)
        * projection to X_1 (the derivative estimate)
        * projection to X_2 (the second derivative estimate) (optional)

    The first two are important within the ODE filter, the latter
    turned out to be very convenient to have.
    """
    def __init__(self, driftmtrx, forcevec, dispmtrx, diffmtrx,
                 ordint, spatialdim):
        """ """
        self.ordint = ordint
        self.spatialdim = spatialdim
        super().__init__(driftmtrx, forcevec, dispmtrx, diffmtrx)

    def proj2coord(self, coord):
        """
        Computes the matrix that projects to the i-th coordinate:
        H_i = I_d \\otimes e_i,
        where e_i is the i-th unit vector.

        Convenience function for development.
        """
        projvec1d = np.eye(self.ordint + 1)[:, coord]
        projmtrx1d = projvec1d.reshape((1, self.ordint + 1))
        return np.kron(np.eye(self.spatialdim), projmtrx1d)


class IBM(ODEPrior):
    """
    IBM(q) (integrated Brownian motion of order q) prior:

    F = I_d \\otimes F
    L = I_d \\otimes L = I_d \\otimes diffconst*(0, ..., 1)
    Q = I_d
    """

    def __init__(self, ordint, spatialdim, diffconst):
        """
        ordint : this is "q"
        spatialdim : d
        diffconst : sigma
        """
        self.diffconst = diffconst
        driftmat = _dynamat_ibm(ordint, spatialdim)
        forcevec = np.zeros(len(driftmat))
        dispvec = _dispvec_ibm_ioup_matern(ordint, spatialdim, diffconst)
        diffmat = np.eye(spatialdim)
        super().__init__(driftmat, forcevec, dispvec, diffmat, ordint, spatialdim)


    def chapmankolmogorov(self, start, stop, step, randvar, *args, **kwargs):
        """
        Overwrites CKE solution with closed form according to IBM.
        The reason is that for this closed form solution here is more
        numerically stable than the matrix exponential.
        "step" variable is obsolent here and is ignored.
        """
        mean, covar = randvar.mean(), randvar.cov()
        ah = self._ah_ibm(start, stop)
        qh = self._qh_ibm(start, stop)
        mpred = ah @ mean
        crosscov = covar @ ah.T
        cpred = ah @ crosscov + qh
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

    def _ah_ibm(self, start, stop):
        """
        Computes A(h)
        """

        def element(stp, rw, cl):
            """Closed form for A(h)_ij"""
            if rw <= cl:
                return stp ** (cl - rw) / np.math.factorial(cl - rw)
            else:
                return 0.0

        step = stop - start
        ah_1d = np.array([[element(step, row, col)
                           for col in range(self.ordint + 1)]
                          for row in range(self.ordint + 1)])
        return np.kron(np.eye(self.spatialdim), ah_1d)

    def _qh_ibm(self, start, stop):
        """
        Computes Q(h)
        """

        def element(stp, ordint, rw, cl, dconst):
            """Closed form for Q(h)_ij"""
            idx = 2 * ordint + 1 - rw - cl
            fact_rw = np.math.factorial(ordint - rw)
            fact_cl = np.math.factorial(ordint - cl)
            return dconst ** 2 * (stp ** idx) / (idx * fact_rw * fact_cl)

        step = stop - start
        qh_1d = np.array([[element(step, self.ordint, row, col, self.diffconst)
                           for col in range(self.ordint + 1)]
                          for row in range(self.ordint + 1)])
        return np.kron(np.eye(self.spatialdim), qh_1d)

def _dynamat_ibm(ordint, spatialdim):
    """
    Returns I_d \\otimes F
    """
    dynamat = np.diag(np.ones(ordint), 1)
    return np.kron(np.eye(spatialdim), dynamat)


class IOUP(ODEPrior):
    """
    IOUP(q) prior:

    F = I_d \\otimes F
    L = I_d \\otimes L = I_d \\otimes (0, ...,  diffconst**2)
    Q = I_d
    """

    def __init__(self, ordint, spatialdim, driftspeed, diffconst):
        """
        ordint : this is "q"
        spatialdim : d
        driftspeed : float > 0; (lambda; note that -lambda ("minus"-lambda)
            is used in the OU equation!!
        diffconst : sigma
        """
        self.driftspeed = driftspeed
        self.diffconst = diffconst
        driftmat = _dynamat_ioup(ordint, spatialdim, self.driftspeed)
        forcevec = np.zeros(len(driftmat))
        dispvec = _dispvec_ibm_ioup_matern(ordint, spatialdim, diffconst)
        diffmat = np.eye(spatialdim)
        super().__init__(driftmat, forcevec, dispvec, diffmat, ordint, spatialdim)


def _dynamat_ioup(ordint, spatialdim, driftspeed):
    """
    Returns I_d \\otimes F
    """
    dynamat = np.diag(np.ones(ordint), 1)
    dynamat[-1, -1] = -driftspeed
    return np.kron(np.eye(spatialdim), dynamat)


class Matern(ODEPrior):
    """
    Matern(q) prior --> Matern process with reg. q+0.5
    and hence, with matrix size q+1

    F = I_d \\otimes F
    L = I_d \\otimes L = I_d \\otimes diffconst*(0, ..., 1)
    Q = I_d
    """

    def __init__(self, ordint, spatialdim, lengthscale, diffconst):
        """
        ordint : this is "q"
        spatialdim : d
        lengthscale : used as 1/lengthscale, remember that!
        diffconst : sigma

        """
        self.lengthscale = lengthscale
        self.diffconst = diffconst
        driftmat = _dynamat_matern(ordint, spatialdim, self.lengthscale)
        forcevec = np.zeros(len(driftmat))
        dispvec = _dispvec_ibm_ioup_matern(ordint, spatialdim, diffconst)
        diffmat = np.eye(spatialdim)
        super().__init__(driftmat, forcevec, dispvec, diffmat, ordint, spatialdim)


def _dynamat_matern(ordint, spatialdim, lengthscale):
    """
    Returns I_d \\otimes F
    """
    dynamat = np.diag(np.ones(ordint), 1)
    nu = ordint + 0.5
    D, lam = ordint + 1,  np.sqrt(2*nu) / lengthscale
    dynamat[-1, :] = np.array([-binom(D, i)*lam**(D-i) for i in range(D)])
    return np.kron(np.eye(spatialdim), dynamat)


def _dispvec_ibm_ioup_matern(ordint, spatialdim, diffconst):
    """
    Returns I_D \otimes L
    diffconst = sigma**2
    """
    dispvec = diffconst * np.eye(ordint + 1)[:, -1]
    return np.kron(np.eye(spatialdim), dispvec).T



