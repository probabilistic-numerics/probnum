"""Integrators, e.g. Integrated Brownian motion, integrated Matern, etc.."""

from probnum.statespace import sde
import numpy as np


class Integrator(sde.LTISDE):
    """
    Integrators are LTI SDEs where each state is a derivative of another state.

    This class has access to things like projection matrices, etc. and is used as a prior
    for ODE solvers.

    They have special structure as $d \otimes q+1$.
    """

    def proj2coord(self, coord: int) -> np.ndarray:
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
        projmat1d_with_precond = projmat @ self.invprecond
        return projmat1d_with_precond


class GeneralMatern(Integrator):
    """With matrices W_0, ..., W_q."""

    pass


class TensorMatern(GeneralMatern):
    """W_i = w_i \otimes I_d"""

    pass


class Matern(TensorMatern):
    """Classical Matern process in 1d."""

    pass


class GeneralIOUP(Integrator):
    """With matrix W."""

    pass


class TensorIOUP(GeneralMatern):
    """W = I_d \\otimes w"""

    pass


class IOUP(TensorMatern):
    """Classical IOUP in 1d"""

    pass


class GeneralIBM(Integrator):
    """Multidimensional IBM"""

    pass


class TensorIBM(Integrator):
    """Multidimensional IBM with tensor product structure."""

    pass


class IBM(TensorIBM):
    """IBM in 1d."""

    pass
