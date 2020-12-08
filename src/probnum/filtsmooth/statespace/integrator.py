"""Integrated processes such as the integrated Wiener process or the Matern process.

This is the sucessor of the former ODEPrior.
"""
try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.special

import probnum.random_variables as pnrv

from . import discrete_transition, sde
from .preconditioner import NordsieckLikeCoordinates


class Integrator:
    """An integrator is a special kind of SDE, where the :math:`i` th coordinate models
    the :math:`i` th derivative."""

    def __init__(self, ordint, spatialdim):
        self.ordint = ordint
        self.spatialdim = spatialdim

    def proj2coord(self, coord: int) -> np.ndarray:
        """Projection matrix to :math:`i` th coordinates.

        Computes the matrix

        .. math:: H_i = \\left[ I_d \\otimes e_i \\right] P^{-1},

        where :math:`e_i` is the :math:`i` th unit vector,
        that projects to the :math:`i` th coordinate of a vector.
        If the ODE is multidimensional, it projects to **each** of the
        :math:`i` th coordinates of each ODE dimension.

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
        return projmat


class IBM(Integrator, sde.LTISDE):
    """Integrated Brownian motion in :math:`d` dimensions."""

    def __init__(self, ordint, spatialdim, diffconst):
        self.diffconst = diffconst

        # initialise BOTH superclasses' inits.
        # I don't like it either, but it does the job.
        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
            self,
            driftmat=self._driftmat,
            forcevec=self._forcevec,
            dispmat=self._dispmat,
        )

        self.precon = NordsieckLikeCoordinates.from_order(ordint, spatialdim)

    @property
    def _driftmat(self):
        driftmat_1d = np.diag(np.ones(self.ordint), 1)
        return np.kron(np.eye(self.spatialdim), driftmat_1d)

    @property
    def _forcevec(self):
        force_1d = np.zeros(self.ordint + 1)
        return np.kron(np.ones(self.spatialdim), force_1d)

    @property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = self.diffconst
        return np.kron(np.eye(self.spatialdim), dispmat_1d).T

    @cached_property
    def equivalent_discretisation_preconditioned(self):
        """Discretised IN THE PRECONDITIONED SPACE."""
        empty_force = np.zeros(self.spatialdim * (self.ordint + 1))
        return discrete_transition.DiscreteLTIGaussian(
            dynamicsmat=self._dynamicsmat,
            forcevec=empty_force,
            diffmat=self._diffmat,
        )

    @cached_property
    def _dynamicsmat(self):
        # Loop, but cached anyway
        driftmat_1d = np.array(
            [
                [
                    scipy.special.binom(self.ordint - i, self.ordint - j)
                    for j in range(0, self.ordint + 1)
                ]
                for i in range(0, self.ordint + 1)
            ]
        )
        return np.kron(np.eye(self.spatialdim), driftmat_1d)

    @cached_property
    def _diffmat(self):
        # Optimised with broadcasting
        range = np.arange(0, self.ordint + 1)
        denominators = 2.0 * self.ordint + 1.0 - range[:, None] - range[None, :]
        diffmat_1d = 1.0 / denominators
        return np.kron(np.eye(self.spatialdim), self.diffconst ** 2 * diffmat_1d)

    def transition_rv(self, rv, start, stop, **kwargs):
        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        step = stop - start
        rv = self.precon.inverse(step) @ rv
        rv, info = self.transition_rv_preconditioned(rv, start)
        info["crosscov"] = self.precon(step) @ info["crosscov"] @ self.precon(step).T
        return self.precon(step) @ rv, info

    def transition_rv_preconditioned(self, rv, start, **kwargs):
        return self.equivalent_discretisation_preconditioned.transition_rv(rv, start)

    def transition_realization(self, real, start, stop, **kwargs):
        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        step = stop - start
        real = self.precon.inverse(step) @ real
        real, info = self.transition_realization_preconditioned(real, start)
        info["crosscov"] = self.precon(step) @ info["crosscov"] @ self.precon(step).T
        return self.precon(step) @ real, info

    def transition_realization_preconditioned(self, real, start, **kwargs):
        return self.equivalent_discretisation_preconditioned.transition_realization(
            real, start
        )

    def discretise(self, step):
        """Equivalent discretisation of the process.

        Overwrites matrix-fraction decomposition in the super-class.
        Only present for user's convenience and to maintain a clean
        interface. Not used for transition_rv, etc..
        """

        dynamicsmat = (
            self.precon(step)
            @ self.equivalent_discretisation_preconditioned.dynamicsmat
            @ self.precon.inverse(step)
        )
        diffmat = (
            self.precon(step)
            @ self.equivalent_discretisation_preconditioned.diffmat
            @ self.precon(step).T
        )
        zero_force = np.zeros(len(dynamicsmat))
        return discrete_transition.DiscreteLTIGaussian(
            dynamicsmat=dynamicsmat, forcevec=zero_force, diffmat=diffmat
        )


class IOUP(Integrator, sde.LTISDE):
    """Integrated Ornstein-Uhlenbeck process in :math:`d` dimensions."""

    def __init__(
        self,
        ordint: int,
        spatialdim: int,
        driftspeed: float,
        diffconst: float,
    ):
        # Other than previously in ProbNum, we do not use preconditioning for IOUP by default.
        self.driftspeed = driftspeed
        self.diffconst = diffconst

        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
            self,
            driftmat=self._driftmat,
            forcevec=self._forcevec,
            dispmat=self._dispmat,
        )

    @property
    def _driftmat(self):
        driftmat_1d = np.diag(np.ones(self.ordint), 1)
        driftmat_1d[-1, -1] = -self.driftspeed
        return np.kron(np.eye(self.spatialdim), driftmat_1d)

    @property
    def _forcevec(self):
        force_1d = np.zeros(self.ordint + 1)
        return np.kron(np.ones(self.spatialdim), force_1d)

    @property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = self.diffconst
        return np.kron(np.eye(self.spatialdim), dispmat_1d).T


class Matern(Integrator, sde.LTISDE):
    """Matern process in :math:`d` dimensions."""

    def __init__(
        self,
        ordint: int,
        spatialdim: int,
        lengthscale: float,
        diffconst: float,
    ):

        # Other than previously in ProbNum, we do not use preconditioning for Matern by default.
        self.lengthscale = lengthscale
        self.diffconst = diffconst

        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
            self,
            driftmat=self._driftmat,
            forcevec=self._forcevec,
            dispmat=self._dispmat,
        )

    @property
    def _driftmat(self):
        driftmat = np.diag(np.ones(self.ordint), 1)
        nu = self.ordint + 0.5
        D, lam = self.ordint + 1, np.sqrt(2 * nu) / self.lengthscale
        driftmat[-1, :] = np.array(
            [-scipy.special.binom(D, i) * lam ** (D - i) for i in range(D)]
        )
        return np.kron(np.eye(self.spatialdim), driftmat)

    @property
    def _forcevec(self):
        force_1d = np.zeros(self.ordint + 1)
        return np.kron(np.ones(self.spatialdim), force_1d)

    @property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = self.diffconst
        return np.kron(np.eye(self.spatialdim), dispmat_1d).T
