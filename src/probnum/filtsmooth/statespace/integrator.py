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
from .preconditioner import TaylorCoordinates


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


class IBM(Integrator, sde.LTISDE):

    preconditioner = TaylorCoordinates

    def __init__(self, ordint, spatialdim, diffconst):
        self.diffconst = diffconst
        F, s, L = self._assemble_ibm_sde(ordint, spatialdim, diffconst)

        # initialise BOTH superclasses' inits.
        # I don't like it either, but it does the job.
        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
            self,
            driftmat=F,
            forcevec=s,
            dispmat=L,
        )

        self.equivalent_discretisation = self.discretise_preconditioned()
        self.precond = self.preconditioner.from_order(
            ordint, spatialdim
        )  # initialise preconditioner class

    @staticmethod
    def _assemble_ibm_sde(ordint, spatialdim, diffconst):
        driftmat_1d = np.diag(np.ones(ordint), 1)
        dispmat_1d = np.zeros(ordint + 1)
        dispmat_1d[-1] = diffconst
        force_1d = np.zeros(ordint + 1)
        I_d = np.eye(spatialdim)
        driftmat, dispmat, force = (
            np.kron(I_d, driftmat_1d),
            np.kron(I_d, dispmat_1d),
            np.kron(np.ones(spatialdim), force_1d),
        )
        return driftmat, force, dispmat

    def discretise_preconditioned(self):
        """Discretised IN THE PRECONDITIONED SPACE."""
        empty_force = np.zeros(self.spatialdim * (self.ordint + 1))
        return discrete_transition.DiscreteLTIGaussian(
            dynamicsmat=self._dynamat,
            forcevec=empty_force,
            diffmat=self._diffmat,
        )

    @cached_property
    def _dynamat(self):
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
            rv, info = self.transition_rv(rv, start, stop, already_preconditioned=True)
            # does the cross-covariance have to be changed somehow??
            return self.precond(step) @ rv, info
        else:
            return self.equivalent_discretisation.transition_rv(rv, start)

    def transition_realization(
        self, real, start, stop, already_preconditioned=False, **kwargs
    ):
        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        step = stop - start
        if not already_preconditioned:
            rv = self.precond.inverse(step) @ real
            rv, info = self.transition_realization(
                rv, start, stop, already_preconditioned=True
            )
            return self.precond(step) @ rv, info
        else:
            return self.equivalent_discretisation.transition_realization(real, start)

    def discretise(self, step):
        """
        Overwrites matrix-fraction decomposition in the super-class.
        Only present for user's convenience and to maintain a clean interface.
        Not used for transition_rv, etc..
        """

        # P and Pinv might have to be swapped...
        dynamicsmatrix = (
            self.precond(step)
            @ self.equivalent_discretisation.dynamicsmat
            @ self.precond.inverse(step)
        )
        diffusionmatrix = (
            self.precond(step)
            @ self.equivalent_discretisation.diffmat
            @ self.precond(step).T
        )
        empty_force = np.zeros(len(dynamicsmatrix))
        return discrete_transition.DiscreteLTIGaussian(
            dynamicsmat=dynamicsmatrix, forcevec=empty_force, diffmat=diffusionmatrix
        )
