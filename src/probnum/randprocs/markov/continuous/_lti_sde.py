"""LTI SDE models as transitions."""
import functools
from typing import Callable, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

from probnum import randvars
from probnum.randprocs.markov import _transition, discrete
from probnum.randprocs.markov.continuous import _mfd
from probnum.typing import FloatArgType, IntArgType
from probnum.utils.linalg import tril_to_positive_tril


class LTISDE(LinearSDE):
    """Linear time-invariant continuous Markov models of the form.

    .. math:: d x(t) = [G x(t) + v] d t + L d w(t).

    In the language of dynamic models,
    x(t) : state process
    G : drift matrix
    v : force term/vector
    L : dispersion matrix.
    w(t) : Wiener process with unit diffusion.

    Parameters
    ----------
    driftmat :
        This is F. It is the drift matrix of the SDE.
    forcevec :
        This is U. It is the force vector of the SDE.
    dispmat :
        This is L. It is the dispersion matrix of the SDE.
    """

    def __init__(
        self,
        driftmat: np.ndarray,
        forcevec: np.ndarray,
        dispmat: np.ndarray,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        _check_initial_state_dimensions(driftmat, forcevec, dispmat)
        dimension = driftmat.shape[0]
        self.driftmat = driftmat
        self.forcevec = forcevec
        self.dispmat = dispmat
        super().__init__(
            dimension,
            (lambda t: self.driftmat),
            (lambda t: self.forcevec),
            (lambda t: self.dispmat),
        )

        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )
        discretised_model = self.discretise(dt=dt)
        return discretised_model.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )
        discretised_model = self.discretise(dt=dt)
        return discretised_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

    @functools.lru_cache(maxsize=None)
    def discretise(self, dt):
        """Return a discrete transition model (i.e. mild solution to SDE) using matrix
        fraction decomposition.

        That is, matrices A(h) and Q(h) and vector s(h) such
        that the transition is

        .. math:: x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h)) ,

        which is the transition of the mild solution to the LTI SDE.
        """

        if np.linalg.norm(self.forcevec) > 0:
            zeros = np.zeros((self.dimension, self.dimension))
            eye = np.eye(self.dimension)
            driftmat = np.block([[self.driftmat, eye], [zeros, zeros]])
            dispmat = np.concatenate((self.dispmat, np.zeros(self.dispmat.shape)))
            ah_stack, qh_stack, _ = _mfd.matrix_fraction_decomposition(
                driftmat, dispmat, dt
            )
            proj = np.eye(self.dimension, 2 * self.dimension)
            proj_rev = np.flip(proj, axis=1)
            ah = proj @ ah_stack @ proj.T
            sh = proj @ ah_stack @ proj_rev.T @ self.forcevec
            qh = proj @ qh_stack @ proj.T
        else:
            ah, qh, _ = _mfd.matrix_fraction_decomposition(
                self.driftmat, self.dispmat, dt
            )
            sh = np.zeros(len(ah))
        return discrete.DiscreteLTIGaussian(
            ah,
            sh,
            qh,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )


def _check_initial_state_dimensions(driftmat, forcevec, dispmat):
    """Checks that the matrices all align and are of proper shape.

    Parameters
    ----------
    driftmat : np.ndarray, shape=(n, n)
    forcevec : np.ndarray, shape=(n,)
    dispmat : np.ndarray, shape=(n, s)
    """
    if driftmat.ndim != 2 or driftmat.shape[0] != driftmat.shape[1]:
        raise ValueError("driftmatrix not of shape (n, n)")
    if forcevec.ndim != 1:
        raise ValueError("force not of shape (n,)")
    if forcevec.shape[0] != driftmat.shape[1]:
        raise ValueError("force not of shape (n,) or driftmatrix not of shape (n, n)")
    if dispmat.ndim != 2:
        raise ValueError("dispersion not of shape (n, s)")
