"""Integrated processes such as the integrated Wiener process or the Matern process.

This is the sucessor of the former ODEPrior.
"""
import functools

import numpy as np
import scipy.special

import probnum.random_variables as pnrv

from . import discrete_transition, sde
from .preconditioner import NordsieckCoordinates


class Integrator:
    """An integrator is a special kind of SDE, which, among other things, has identifyable coordinates."""

    def __init__(self, ordint, spatialdim):
        self.ordint = ordint
        self.spatialdim = spatialdim

    def proj2deriv(self, coord: int) -> np.ndarray:
        """
        Projection matrix to :math:`i`-th coordinates.

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
        return projmat


class IBM(sde.LTISDE, Integrator):

    preconditioner = NordsieckCoordinates

    def __init__(self, ordint, spatialdim, diffconst):
        self.diffconst = diffconst
        F, s, L = self._assemble_ibm_sde(ordint, spatialdim, diffconst)

        # initialise BOTH superclasses' inits.
        # I don't like it either, but it does the job.
        super().__init__(
            driftmatrix=F,
            forcevec=s,
            dispmatrix=L,
            ordint=ordint,
            spatialdim=spatialdim,
        )

        self.equivalent_discretisation = self.discretise()
        self.precond = self.preconditioner.from_order(
            ordint, spatialdim
        )  # initialise preconditioner class

    @staticmethod
    def _assemble_ibm_sde(self, ordint, spatialdim, diffconst):
        F_1d = np.diag(np.ones(ordint), 1)
        L_1d = np.zeros(ordint + 1)
        L_1d[-1] = diffconst
        s_1d = np.np.zeros(ordint + 1)
        I_d = np.eye(spatialdim)
        F, L, s = np.kron(F_1d, I_d), np.kron(L_1d, I_d), np.kron(s_1d, I_d)
        return F, s, L

    def discretise_preconditioned(self):
        """Discretised IN THE PRECONDITIONED SPACE."""
        ah = self._driftmatrix
        qh = self._diffusionmatrix
        empty_force = np.zeros(len(ah))
        return discrete_transition.DiscreteLTIGaussian(
            driftmatrix=self._driftmatrix,
            forcevec=empty_force,
            diffusionmatrix=self._diffusionmatrix,
        )

    @functools.cached_property
    def _driftmatrix(self):
        # Loop, but cached anyway
        drifmat_1d = np.array(
            [
                [
                    scipy.special.binom(self.ordint - i, self.ordint - j)
                    for j in range(0, self.ordint + 1)
                ]
                for i in range(0, self.ordint + 1)
            ]
        )
        return np.kron(drifmat_1d, np.eye(self.spatialdim))

    @functools.cached_property
    def _diffusionmatrix(self):
        # Optimised with broadcaseing
        range = np.arange(0, self.ordint + 1)
        denominators = 2.0 * self.ordint + 1.0 - range[:, None] - range[None, :]
        diffmat_1d = 1.0 / denominators
        return np.kron(diffmat_1d, np.eye(self.spatialdim))

    def transition_rv(self, rv, start, stop, already_preconditioned=False, **kwargs):
        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        step = stop - start
        if not already_preconditioned:
            rv = self.precond.inverse(step) @ rv
            rv = self.transition_rv(rv, start, stop, already_preconditioned=True)
            return self.precond(step) @ rv
        else:
            return self.equivalent_discretisation.transition_rv(rv)

    def transition_realization(
        self, real, start, stop, already_preconditioned=False, **kwargs
    ):
        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        step = stop - start
        if not already_preconditioned:
            rv = self.precond.inverse(step) @ real
            rv = self.transition_realization(
                rv, start, stop, already_preconditioned=True
            )
            return self.precond.inverse(step) @ rv
        else:
            return self.equivalent_discretisation.transition_realization(real)

    def discretise(self, step):
        """
        Overwrites matrix-fraction decomposition in the super-class.
        Only present for user's convenience and to maintain a clean interface.
        Not used for transition_rv, etc..
        """

        # P and Pinv might have to be swapped...
        dynamicsmatrix = (
            self.precond(step)
            @ self.equivalent_discretisation.dynamicsmatrix
            @ self.precond.inverse(step)
        )
        diffusionmatrix = (
            self.precond(step)
            @ self.equivalent_discretisation.diffusionmatrix
            @ self.precond(step).T
        )
        empty_force = np.zeros(len(dynamicsmatrix))
        return discrete_transition.DiscreteLTIGaussian(
            dynamat=dynamicsmatrix, forcevec=empty_force, diffmat=diffusionmatrix
        )
