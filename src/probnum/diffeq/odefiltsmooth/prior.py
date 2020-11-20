"""Continuous-Time priors for ODE solvers.

Currently, they are only relevant in the context of ODEs. If needed in a
more general setting, it is easy to move them to statespace module
"""
import warnings

import numpy as np
from scipy.special import binom  # for Matern
from scipy.special import factorial  # vectorised factorial for IBM-Q(h)

import probnum.filtsmooth.statespace as pnfss


class ODEPrior(pnfss.LTISDE):
    """Prior dynamic model for ODE filtering and smoothing.

    An ODE prior is an continuous LTI state space model with attributes:
        * order of integration :math:`q`
        * spatial dimension of the underlying ODE
        * projection to :math:`X_i(t)`
          (the :math:`(i-1)`-th derivative estimate)
        * A preconditioner :math:`P` (see below)

    Instead of the LTI SDE

    .. math:: d X(t) = [F X(t) + u] dt + L dB(t)

    the prior for the ODE Dynamics is given by

    .. math:: dX(t) = P F P^{-1} X(t) dt + P L dB(t)

    where :math:`P` is a preconditioner matrix ensuring stability
    of the iterations. Note that ODE priors do not have a drift term.
    By default, we choose :math:`P` to be the
    matrix that maps to filtering iteration to the Nordsieck vector,

    .. math:: P = \\text{diag }(h^{-q}, h^{-q+1}, ..., 1).

    Here, :math:`h` is some expected average step size. Note that we
    ignored the factorials in this matrix. Our setting makes it easy to
    recover "no preconditioning" by choosing :math:`h=1`.

    - If no expected step size is available we choose :math:`h=1.0`.
      This recovers :math:`P=I_{d(q+1)}`, hence no preconditioning.

    - For fixed step size algorithms this quantity :math:`h` is easy
      to choose

    - For adaptive steps it is a bit more involved.

    Since it doesn't have to be exact, any more or less appropriate
    choice will do well. The main effect of this preconditioning is that
    the predictive covariances inside each filter iteration are
    well-conditioned: for IBM(:math:`q`) priors, the condition number
    of the predictive covariances only depends on order of integration
    :math:`q` and not on the step size anymore. Nb: this only holds
    if all required derivatives of the RHS vector field of the ODE are
    specified: None for IBM(1), Jacobian of :math:`f` for IBM(2),
    Hessian of :math:`f` for IBM(3). If this is not the case the
    preconditioner still helps but is not as powerful anymore.

    Without preconditioning they can be numerically singular for small
    steps and higher order methods which especially makes smoothing
    algorithms unstable.

    Another advantage of this preconditioning is that the smallest value
    that appears inside the algorithm is :math:`h^{q}` (with
    preconditioning) instead of :math:`h^{2q+1}` (without preconditioning).

    The matrices :math:`F, u, L` are the usual matrices for
    IBM(:math:`q`), IOUP(:math:`q`) or Matern(:math:`q+1/2`) processes.
    As always, :math:`B(t)` is
    s-dimensional Brownian motion with unit diffusion matrix :math:`Q`.
    """

    def __init__(
        self,
        driftmat: np.ndarray,
        dispmat: np.ndarray,
        ordint: int,
        spatialdim: int,
        precond_step: float = 1.0,
    ):
        self.ordint = ordint
        self.spatialdim = spatialdim
        self.precond, self.invprecond = self.precond2nordsieck(precond_step)
        driftmat = self.precond @ driftmat @ self.invprecond
        dispmat = self.precond @ dispmat
        forcevec = np.zeros(len(driftmat))
        super().__init__(driftmat, forcevec, dispmat)

    def proj2coord(self, coord: int) -> np.ndarray:
        """Projection matrix to :math:`i`-th coordinates.

        Computes the matrix

        .. math:: H_i = \\left[ I_d \\otimes e_i \\right] P^{-1},

        where :math:`e_i` is the :math:`i`-th unit vector,
        that projects to the :math:`i`-th coordinate of a vector.
        If the ODE is multidimensional, it projects to **each** of the
        :math:`i`-th coordinates of each ODE dimension.

        Parameters
        ----------
        coord : int
            Coordinate index :math:`i` which to project to.
            Expected to be in range :math:`0 \\leq i \\leq q + 1`.

        Returns
        -------
        np.ndarray, shape=(d, d*(q+1))
            Projection matrix :math:`H_i`.
        """
        projvec1d = np.eye(self.ordint + 1)[:, coord]
        projmat1d = projvec1d.reshape((1, self.ordint + 1))
        projmat = np.kron(np.eye(self.spatialdim), projmat1d)
        projmat1d_with_precond = projmat @ self.invprecond
        return projmat1d_with_precond

    def precond2nordsieck(self, step: float) -> (np.ndarray, np.ndarray):
        """Computes preconditioner inspired by Nordsieck.

        Computes the matrix :math:`P` given by

        .. math:: P = I_d \\otimes diag (1, h, h^2, ..., h^q)

        as well as its inverse :math:`P^{-1}`.

        Parameters
        ----------
        step : float
            Step size :math:`h` used for preconditioning. If :math:`h`
            is so small that :math:`h^q! < 10^{-15}`, it is being
            set to :math:`h = (\\cdot 10^{-15})^{1/q}`.

        Returns
        -------
        precond : np.ndarray, shape=(d(q+1), d(q+1))
            Preconditioner matrix :math:`P`.
        invprecond : np.ndarray, shape=(d(q+1), d(q+1))
            Inverse preconditioner matrix :math:`P^{-1}`.
        """
        smallval = step ** self.ordint
        if smallval < 1e-15:
            warnmsg = (
                "Preconditioner contains values below "
                "machine precision (%.1e)" % smallval
            )
            warnings.warn(message=warnmsg, category=RuntimeWarning)
            step = 1e-15 ** (1 / self.ordint)
        powers = np.arange(start=-self.ordint, stop=1)
        diags = step ** powers
        precond = np.kron(np.eye(self.spatialdim), np.diag(diags))
        invprecond = np.kron(np.eye(self.spatialdim), np.diag(1.0 / diags))
        return precond, invprecond

    @property
    def preconditioner(self) -> np.ndarray:
        """Convenience property to return the readily-computed preconditioner without
        having to remember abbreviations.

        Returns
        -------
        np.ndarray, shape=(d(q+1), d(q+1))
            Preconditioner matrix :math:`P`
        """
        return self.precond

    @property
    def inverse_preconditioner(self) -> np.ndarray:
        """Convenience property to return the readily-computed inverse preconditioner
        without having to remember abbreviations.

        Returns
        -------
        np.ndarray, shape=(d(q+1), d(q+1))
            Inverse preconditioner matrix :math:`P^{-1}`
        """
        return self.invprecond


class IBM(ODEPrior):
    """Integrated Brownian motion of order :math:`q` prior.

    The integrated Brownian motion prior is represented through
    the LTI SDE

    .. math:: dX(t) =  F X(t) dt + L dB(t)

    where for readibility reasons we did not write the preconditioner
    matrix :math:`P`; see :class:`ODEPrior` for explanations.

    - It has driftmatrix :math:`F` given by

      .. math:: F = I_d \\otimes \\tilde F, \\quad
        \\tilde F = \\begin{pmatrix} 0 & I_q \\\\ 0 & 0 \\end{pmatrix}

      where the top left zero-vector has :math:`q` rows and 1 column.

    - It has dispersion matrix :math:`L` given by

      .. math:: L = I_d \\otimes \\tilde L, \\quad
        \\tilde L = \\sigma \\, e_{q+1}

      where :math:`\\sigma` is the diffusion constant, that is,
      :math:`\\sigma^2` is the intensity of each dimension of the
      :math:`d`-dimensional Brownian motion driving the SDE and
      :math:`e_{q+1}=(0, ..., 0, 1)` is the :math:`(q+1)`-st unit vector.

    - The Brownian motion :math:`B=B(t)` driving the SDE has unit
      diffusion :math:`Q = I_d`.

    Parameters
    ----------
    ordint :
        Order of integration :math:`q`. The higher :math:`q`, the higher
        the order of the ODE filter.
    spatialdim :
        Spatial dimension :math:`d` of the ordinary differential
        equation that is to be modelled.
    diffconst :
        Diffusion constant :math:`sigma` of the stochastic process.
    precond_step :
        Expected step size :math:`h` used in the ODE filter.
        This quantity is used for preconditioning, see :class:`ODEPrior`
        for a clear explanation. Default is :math:`h=1`.
    """

    def __init__(
        self, ordint: int, spatialdim: int, diffconst: float, precond_step: float = 1.0
    ) -> None:
        self.diffconst = diffconst
        driftmat = _driftmat_ibm(ordint, spatialdim)
        dispmat = _dispmat(ordint, spatialdim, diffconst)
        super().__init__(driftmat, dispmat, ordint, spatialdim, precond_step)

    def _discretise(self, step: float) -> (np.ndarray, np.ndarray, np.ndarray):
        dynamicsmatrix = self._trans_ibm(step)
        empty_force = np.zeros(len(dynamicsmatrix))
        diffusionmatrix = self._transdiff_ibm(step)
        return dynamicsmatrix, empty_force, diffusionmatrix

    def _trans_ibm(self, step: float) -> np.ndarray:
        """Computes closed form solution for the transition matrix A(h)."""

        # This seems like the faster solution compared to fully vectorising.
        # I suspect it is because np.math.factorial is much faster than
        # scipy.special.factorial
        ah_1d = np.diag(np.ones(self.ordint + 1), 0)
        for i in range(self.ordint):
            offdiagonal = (
                step ** (i + 1) / np.math.factorial(i + 1) * np.ones(self.ordint - i)
            )
            ah_1d += np.diag(offdiagonal, i + 1)

        ah = np.kron(np.eye(self.spatialdim), ah_1d)
        return self.precond @ ah @ self.invprecond

    def _transdiff_ibm(self, step: float) -> np.ndarray:
        """Computes closed form solution for the diffusion matrix Q(h)."""
        indices = np.arange(0, self.ordint + 1)
        col_idcs, row_idcs = np.meshgrid(indices, indices)
        exponent = 2 * self.ordint + 1 - row_idcs - col_idcs
        factorial_rows = factorial(
            self.ordint - row_idcs
        )  # factorial() handles matrices but is slow(ish)
        factorial_cols = factorial(self.ordint - col_idcs)
        qh_1d = (
            self.diffconst ** 2
            * step ** exponent
            / (exponent * factorial_rows * factorial_cols)
        )
        qh = np.kron(np.eye(self.spatialdim), qh_1d)
        return self.precond @ qh @ self.precond.T


def _driftmat_ibm(ordint: int, spatialdim: int) -> np.ndarray:
    """Returns I_d \\otimes F."""
    driftmat_1d = np.diag(np.ones(ordint), 1)
    return np.kron(np.eye(spatialdim), driftmat_1d)


class IOUP(ODEPrior):
    """IOUP(q) prior:

    F = I_d \\otimes F
    L = I_d \\otimes L = I_d \\otimes (0, ...,  diffconst**2)
    Q = I_d

    Parameters
    ----------
    ordint :
        this is "q"
    spatialdim :
        d
    driftspeed :
        float > 0; (lambda; note that -lambda ("minus"-lambda)
        is used in the OU equation!!
    diffconst :
        sigma
    """

    def __init__(
        self,
        ordint: int,
        spatialdim: int,
        driftspeed: float,
        diffconst: float,
        precond_step: float = 1.0,
    ):
        self.driftspeed = driftspeed
        self.diffconst = diffconst
        driftmat = _driftmat_ioup(ordint, spatialdim, self.driftspeed)
        dispvec = _dispmat(ordint, spatialdim, diffconst)
        super().__init__(driftmat, dispvec, ordint, spatialdim, precond_step)


def _driftmat_ioup(ordint: int, spatialdim: int, driftspeed: float) -> np.ndarray:
    """Return :math:`I_d \\otimes F`."""
    driftmat = np.diag(np.ones(ordint), 1)
    driftmat[-1, -1] = -driftspeed
    return np.kron(np.eye(spatialdim), driftmat)


class Matern(ODEPrior):
    """Matern(q) prior --> Matern process with reg. q+0.5 and hence, with matrix size
    q+1.

    F = I_d \\otimes F
    L = I_d \\otimes L = I_d \\otimes diffconst*(0, ..., 1)
    Q = I_d

    Parameters
    ----------
    ordint :
        this is "q"
    spatialdim :
        d
    lengthscale :
        used as 1/lengthscale, remember that!
    diffconst :
        sigma
    """

    def __init__(
        self,
        ordint: int,
        spatialdim: int,
        lengthscale: float,
        diffconst: float,
        precond_step: float = 1.0,
    ):
        self.lengthscale = lengthscale
        self.diffconst = diffconst
        driftmat = _driftmat_matern(ordint, spatialdim, self.lengthscale)
        dispvec = _dispmat(ordint, spatialdim, diffconst)
        super().__init__(driftmat, dispvec, ordint, spatialdim, precond_step)


def _driftmat_matern(ordint: int, spatialdim: int, lengthscale: float) -> np.ndarray:
    """Returns I_d \\otimes F."""
    driftmat = np.diag(np.ones(ordint), 1)
    nu = ordint + 0.5
    D, lam = ordint + 1, np.sqrt(2 * nu) / lengthscale
    driftmat[-1, :] = np.array([-binom(D, i) * lam ** (D - i) for i in range(D)])
    return np.kron(np.eye(spatialdim), driftmat)


def _dispmat(ordint: int, spatialdim: int, diffconst: float) -> np.ndarray:
    """Returns I_D \\otimes L
    diffconst = sigma**2
    """
    dispvec = diffconst * np.eye(ordint + 1)[:, -1]
    return np.kron(np.eye(spatialdim), dispvec).T
