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
import scipy.linalg  # for pascal (precond. A(h))

from probnum.filtsmooth.statespace.continuous import LTISDEModel
from probnum.prob import RandomVariable, Normal

class ODEPrior(LTISDEModel):
    """
    Prior dynamic model for ODE filtering and smoothing.

    Instead of the LTI SDE

    .. math:: d X(t) = [F X(t) + u] dt + L dB(t)

    the prior for the ODE Dynamics is given by

    .. math:: dX(t) = [P F P^{-1} X(t) + P u] dt + P L dB(t)

    where :math:`P` is a preconditioner matrix ensuring stability
    of the iterations. By default, we choose :math:`P` to be the
    matrix that maps to filtering iteration to the Nordsieck vector,

    .. math:: P = \\text{diag }(1, h, h^2/2, ..., h^q/q!).
    
    Here, :math:`h` is some expected average step size. For fixed step
    algorithms this is easy to choose, and for adaptive steps it is a
    bit more involved. Since it doesn't have to be exact, any decent
    choice will do well. The main effect of this preconditioning is that
    the predictive covariances inside each filter iteration are
    well-conditioned.
    Without preconditioning they can be numerically singular for small
    steps and higher order methods which makes smoothing instable.

    The matrices :math:`F, u, L` are the usual matrices for
    IBM(:math:`q`), IOUP(:math:`q`) or Matern(:math:`q+1/2`) processes.
    As always, :math:`B(t)` is
    s-dimensional Brownian motion with diffusion matrix :math:`Q`.

    An ODE prior is an LTI state space model with specific attributes:
        * order of integration :math:`q`
        * spatial dimension of the underlying ODE
        * projection to :math:`X_i(t)` (the :math:`(i-1)`-th derivative estimate)
        * A preconditioner :math:`P`

    The first two are important within the ODE filter, the latter
    turned out to be very convenient to have.
    """
    def __init__(self, driftmat, forcevec, dispmat, diffmat,
                 ordint, spatialdim, precond=False, expectedstep=None):
        """ """
        self.ordint = ordint
        self.spatialdim = spatialdim
        if precond is True:
            if expectedstep is None:
                raise TypeError("Please provide an expected step size "
                                "for preconditioning.")
            self.precond = self.precond2nordsieck(expectedstep)
            self.invprecond = np.linalg.inv(self.precond)  # stable bc. diag.
            driftmat = self.precond @ driftmat @ self.invprecond
            forcevec = self.precond @ forcevec
            dispmat = self.precond @ dispmat
        else:
            self.precond = None
            self.invprecond = None
        super().__init__(driftmat, forcevec, dispmat, diffmat)

    def proj2coord(self, coord):
        """
        Computes the matrix that projects to the i-th coordinate:
        :math:`H_i = I_d \\otimes e_i`,
        where :math:`e_i` is the :math:`i`-th unit vector.

        If preconditioning is desired, the projection matrices
        become :math:`H_i P^{-1}`.
        """
        projvec1d = np.eye(self.ordint + 1)[:, coord]
        projmat1d = projvec1d.reshape((1, self.ordint + 1))
        if self.invprecond is not None:
            projmat1d = projmat1d @ self.invprecond
        return np.kron(np.eye(self.spatialdim), projmat1d)

    def precond2nordsieck(self, step):
        """
        Makes preconditioner from Eq. (31) in Schober et al.

        .. math:: P = I_d \\otimes diag (1, h, h^2/2, ..., h^q/q!)

        """
        smallval = step**self.ordint / np.math.factorial(self.ordint)
        if smallval < 1e-15:
            print("Warning: preconditioner contains values below "
                  "machine precision", smallval)
            step = 1e-15**(1/self.ordint)
        powers = np.arange(self.ordint + 1)
        factorials = np.array([np.math.factorial(pw) for pw in powers])
        diags = step**powers / factorials
        precond = np.kron(np.eye(self.spatialdim), np.diag(diags))
        return precond

    @property
    def preconditioner(self):
        """ """
        return self.precond

    @property
    def inverse_preconditioner(self):
        """ """
        return self.invprecond




class IBM(ODEPrior):
    """
    IBM(q) (integrated Brownian motion of order q) prior:

    F = I_d \\otimes F
    L = I_d \\otimes L = I_d \\otimes diffconst*(0, ..., 1)
    Q = I_d
    """

    def __init__(self, ordint, spatialdim, diffconst, precond=False, expectedstep=None):
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
        super().__init__(driftmat, forcevec, dispvec, diffmat, ordint, spatialdim, precond, expectedstep)


    def chapmankolmogorov(self, start, stop, step, randvar, *args, **kwargs):
        """
        Overwrites CKE solution with closed form according to IBM.

        This more stable than the matrix-exponential implementation
        in super().chapmankolmogorov(...) which is relevant for
        higher order IBM priors and smaller steps---in these cases the
        preconditioning does not do as much as it should do without
        overwriting.

        "step" variable is obsolent here and is ignored.
        """
        mean, covar = randvar.mean(), randvar.cov()
        ah = self._ah_ibm(start, stop)
        qh = self._qh_ibm(start, stop)
        mpred = ah @ mean
        crosscov = covar @ ah.T
        cpred = ah @ crosscov + qh
        # print("Just Q(h):\n", qh, np.linalg.cond(qh))
        # print("Predicted covariance:\n", cpred, np.linalg.cond(cpred))
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
        if self.precond is not None:
            ah_1d = scipy.linalg.pascal(self.ordint + 1, "upper")
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
        if self.precond is not None:
            qh_1d = self.precond @ qh_1d @ self.precond.T
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

    def __init__(self, ordint, spatialdim, driftspeed, diffconst, precond=False, expectedstep=None):
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
        super().__init__(driftmat, forcevec, dispvec, diffmat, ordint, spatialdim, precond, expectedstep)


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

    def __init__(self, ordint, spatialdim, lengthscale, diffconst, precond=False, expectedstep=None):
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
        super().__init__(driftmat, forcevec, dispvec, diffmat, ordint, spatialdim, precond, expectedstep)


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



