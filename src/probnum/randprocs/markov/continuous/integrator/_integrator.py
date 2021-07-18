"""Integrated processes such as the integrated Wiener process or the Matern process.

This is the sucessor of the former ODEPrior.
"""

import numpy as np

from probnum.randprocs.markov.continuous.integrator import _preconditioner
from probnum.typing import IntArgType

__all__ = ["IntegratorTransition"]


class IntegratorTransition:
    """Integrator processes.

    An integrator is a special kind of random process
    that models a stack of a state and its first :math:`\nu` time-derivatives.
    For instances, this includes integrated Wiener processes or Matern processes.

    In ProbNum, integrators are always derived by driving :math:`d` dimensional Wiener processes.
    We compute the transitions usually in Nordsieck-Coordinates.
    """

    def __init__(self, nu, wiener_process_dimension):
        self.nu = nu
        self.wiener_process_dimension = wiener_process_dimension
        self.precon = _preconditioner.NordsieckLikeCoordinates.from_order(
            nu, wiener_process_dimension
        )

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
        projvec1d = np.eye(self.nu + 1)[:, coord]
        projmat1d = projvec1d.reshape((1, self.nu + 1))
        projmat = np.kron(np.eye(self.wiener_process_dimension), projmat1d)
        return projmat

    @property
    def _derivwise2coordwise_projmat(self) -> np.ndarray:
        r"""Projection matrix to change the ordering of the state representation in an :class:`Integrator` from coordinate-wise to derivative-wise representation.

        A coordinate-wise ordering is

        .. math:: (y_1, \dot y_1, \ddot y_1, y_2, \dot y_2, ..., y_d^{(\nu)})

        and a derivative-wise ordering is

        .. math:: (y_1, y_2, ..., y_d, \dot y_1, \dot y_2, ..., \dot y_d, \ddot y_1, ..., y_d^{(\nu)}).

        Default representation in an :class:`Integrator` is coordinate-wise ordering, but sometimes, derivative-wise ordering is more convenient.

        See Also
        --------
        :attr:`Integrator._convert_coordwise_to_derivwise`
        :attr:`Integrator._convert_derivwise_to_coordwise`

        """
        dim = (self.nu + 1) * self.wiener_process_dimension
        projmat = np.zeros((dim, dim))
        E = np.eye(dim)
        for q in range(self.nu + 1):

            projmat[q :: (self.nu + 1)] = E[
                q
                * self.wiener_process_dimension : (q + 1)
                * self.wiener_process_dimension
            ]
        return projmat

    @property
    def _coordwise2derivwise_projmat(self) -> np.ndarray:
        r"""Projection matrix to change the ordering of the state representation in an :class:`Integrator` from derivative-wise to coordinate-wise representation.

        A coordinate-wise ordering is

        .. math:: (y_1, \dot y_1, \ddot y_1, y_2, \dot y_2, ..., y_d^{(\nu)})

        and a derivative-wise ordering is

        .. math:: (y_1, y_2, ..., y_d, \dot y_1, \dot y_2, ..., \dot y_d, \ddot y_1, ..., y_d^{(\nu)}).

        Default representation in an :class:`Integrator` is coordinate-wise ordering, but sometimes, derivative-wise ordering is more convenient.

        See Also
        --------
        :attr:`Integrator._convert_coordwise_to_derivwise`
        :attr:`Integrator._convert_derivwise_to_coordwise`

        """
        return self._derivwise2coordwise_projmat.T
