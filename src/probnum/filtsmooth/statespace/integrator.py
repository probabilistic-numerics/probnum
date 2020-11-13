"""Integrated processes such as the integrated Wiener process or the Matern process.

This is the sucessor of the former ODEPrior.
"""
import numpy as np
import scipy.special

import probnum.random_variabls as pnrv

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
        F, s, L = self._assemble_ibm_sde()

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

    def _assemble_ibm_sde(self):
        F_1d = np.diag(np.ones(ordint), 1)
        L_1d = np.zeros(ordint + 1)
        L_1d[-1] = diffconst
        s_1d = np.np.zeros(ordint + 1)
        I_d = np.eye(spatialdim)
        F, L, s = np.kron(F_1d, I_d), np.kron(L_1d, I_d), np.kron(s_1d, I_d)
        return F, s, L

    def discretise_preconditioned(self):
        """Discretised IN THE PRECONDITIONED SPACE."""
        raise NotImplementedError("Do this nico!")

    def discretise(self, step):
        """Overwrite MFD. Only present for user's convenience and to maintain a clean interface."""
        dynamicsmatrix = self._trans_ibm(step)
        empty_force = np.zeros(len(dynamicsmatrix))
        diffusionmatrix = self._transdiff_ibm(step)
        return discrete_transition.DiscreteLTIGaussian(
            dynamat=dynamicsmatrix, forcevec=empty_force, diffmat=diffusionmatrix
        )

    def _trans_ibm(self, step: float) -> np.ndarray:
        """
        Computes closed form solution for the transition matrix A(h).
        """

        # This seems like the faster solution compared to fully vectorising.
        # I suspect it is because np.math.factorial is much faster than
        # scipy.special.factorial
        ah_1d = np.diag(np.ones(self.ordint + 1), 0)
        for i in range(self.ordint):
            offdiagonal = (
                step ** (i + 1) / np.math.factorial(i + 1) * np.ones(self.ordint - i)
            )
            ah_1d += np.diag(offdiagonal, i + 1)

        return np.kron(np.eye(self.spatialdim), ah_1d)

    def _transdiff_ibm(self, step: float) -> np.ndarray:
        """
        Computes closed form solution for the diffusion matrix Q(h).
        """
        indices = np.arange(0, self.ordint + 1)
        col_idcs, row_idcs = np.meshgrid(indices, indices)
        exponent = 2 * self.ordint + 1 - row_idcs - col_idcs
        factorial_rows = scipy.special.factorial(
            self.ordint - row_idcs
        )  # factorial() handles matrices but is slow(ish)
        factorial_cols = scipy.special.factorial(self.ordint - col_idcs)
        qh_1d = (
            self.diffconst ** 2
            * step ** exponent
            / (exponent * factorial_rows * factorial_cols)
        )
        return np.kron(np.eye(self.spatialdim), qh_1d)

    def transition_rv(self, rv, start, stop, already_preconditioned=False, **kwargs):
        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        step = stop - start
        if not already_preconditioned:
            rv = self.precond(step) @ rv
            rv = self.transition_rv(rv, start, stop, already_preconditioned=True)
            return self.precond.inverse(step) @ rv
        else:
            return self.equivalent_discretisation.transition_rv(rv)

    def transition_realization(self, real, start, stop, already_preconditioned=False, **kwargs):
        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        step = stop - start
        if not already_preconditioned:
            rv = self.preconditioner(step) @ real
            rv = self.transition_realization(
                rv, start, stop, already_preconditioned=True
            )
            return self.precond.inverse(step) @ rv
        else:
            return self.equivalent_discretisation.transition_realization(real)
