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

import probnum.typing as pntype
from probnum import randvars
from probnum.randprocs.markov import discrete
from probnum.randprocs.markov.continuous import _sde
from probnum.randprocs.markov.continuous.integrator import _preconditioner


class Integrator:
    """An integrator is a special kind of SDE, where the :math:`i` th coordinate models
    the :math:`i` th derivative."""

    def __init__(self, ordint, spatialdim):
        self.ordint = ordint
        self.spatialdim = spatialdim
        self.precon = _preconditioner.NordsieckLikeCoordinates.from_order(
            self.ordint, self.spatialdim
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
        projvec1d = np.eye(self.ordint + 1)[:, coord]
        projmat1d = projvec1d.reshape((1, self.ordint + 1))
        projmat = np.kron(np.eye(self.spatialdim), projmat1d)
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
        dim = (self.ordint + 1) * self.spatialdim
        projmat = np.zeros((dim, dim))
        E = np.eye(dim)
        for q in range(self.ordint + 1):

            projmat[q :: (self.ordint + 1)] = E[
                q * self.spatialdim : (q + 1) * self.spatialdim
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

    @staticmethod
    def _convert_coordwise_to_derivwise(
        state: np.ndarray, ordint: pntype.IntArgType, spatialdim: pntype.IntArgType
    ) -> np.ndarray:
        """Convert coordinate-wise representation to derivative-wise representation.

        Lightweight call to the respective property in :class:`Integrator`.

        Parameters
        ----------
        state:
            State to be converted. Assumed to be in coordinate-wise representation.
        ordint:
            Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
        spatialdim:
            Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.

        See Also
        --------
        :attr:`Integrator._coordwise2derivwise_projmat`
        :attr:`Integrator._derivwise2coordwise_projmat`
        """
        projmat = Integrator(ordint, spatialdim)._coordwise2derivwise_projmat
        return projmat @ state

    @staticmethod
    def _convert_derivwise_to_coordwise(
        state: np.ndarray, ordint: pntype.IntArgType, spatialdim: pntype.IntArgType
    ) -> np.ndarray:
        """Convert coordinate-wise representation to derivative-wise representation.

        Lightweight call to the respective property in :class:`Integrator`.

        Parameters
        ----------
        state:
            State to be converted. Assumed to be in derivative-wise representation.
        ordint:
            Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
        spatialdim:
            Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.


        See Also
        --------
        :attr:`Integrator._coordwise2derivwise_projmat`
        :attr:`Integrator._derivwise2coordwise_projmat`
        """
        projmat = Integrator(ordint, spatialdim)._derivwise2coordwise_projmat
        return projmat @ state
