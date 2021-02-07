"""SDE models as transitions."""
import functools
from typing import Callable, Optional

import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import discrete_transition, transition
from .sde_utils import linear_sde_statistics, matrix_fraction_decomposition


class SDE(transition.Transition):
    """Stochastic differential equation.

    .. math:: d x(t) = g(t, x(t)) d t + L(t) d w(t),

    driven by a Wiener process with unit diffusion.
    """

    def __init__(
        self,
        driftfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        dispmatfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        jacobfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        dimension=None,
    ):
        self.driftfun = driftfun
        self.dispmatfun = dispmatfun
        self.jacobfun = jacobfun
        super().__init__(input_dim=dimension, output_dim=dimension)

    def forward_realization(
        self, real, t, dt=None, _compute_gain=False, _diffusion=1.0
    ):
        raise NotImplementedError

    def forward_rv(self, rv, t, dt=None, _compute_gain=False, _diffusion=1.0):
        raise NotImplementedError

    def backward_realization(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
    ):
        raise NotImplementedError

    def backward_rv(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
    ):
        raise NotImplementedError


class LinearSDE(SDE):
    """Linear stochastic differential equation (SDE),

    .. math:: d x(t) = [G(t) x(t) + v(t)] d t + L(t) x(t) d w(t).

    For Gaussian initial conditions, this solution is a Gaussian process.

    Parameters
    ----------
    driftmatfun :
        This is G = G(t). The evaluations of this function are called
        the driftmatrix of the SDE.
        Returns np.ndarray with shape=(n, n)
    forcevecfun :
        This is v = v(t). Evaluations of this function are called
        the force(vector) of the SDE.
        Returns np.ndarray with shape=(n,)
    dispmatfun :
        This is L = L(t). Evaluations of this function are called
        the dispersion(matrix) of the SDE.
        Returns np.ndarray with shape=(n, s)
    """

    def __init__(
        self,
        driftmatfun: Callable[[FloatArgType], np.ndarray],
        forcevecfun: Callable[[FloatArgType], np.ndarray],
        dispmatfun: Callable[[FloatArgType], np.ndarray],
        dimension=None,
    ):
        self.driftmatfun = driftmatfun
        self.forcevecfun = forcevecfun
        super().__init__(
            driftfun=(lambda t, x: driftmatfun(t) @ x + forcevecfun(t)),
            dispmatfun=dispmatfun,
            jacobfun=(lambda t, x: driftmatfun(t)),
            dimension=dimension,
        )

    def forward_realization(
        self,
        real,
        start,
        stop,
        step,
        _diffusion=1.0,
        **kwargs,
    ):

        rv = pnrv.Normal(real, 0 * np.eye(len(real)))
        return linear_sde_statistics(
            rv,
            start,
            stop,
            step,
            self.driftfun,
            self.driftmatfun,
            self.dispmatfun,
            _diffusion=_diffusion,
        )

    def forward_rv(self, rv, start, stop, step, _diffusion=1.0, **kwargs):

        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in linear SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        return linear_sde_statistics(
            rv,
            start,
            stop,
            step,
            self.driftfun,
            self.driftmatfun,
            self.dispmatfun,
            _diffusion=_diffusion,
        )

    def backward_realization(
        self,
        real,
        rv_past,
        start,
        stop=None,
        step=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        raise NotImplementedError

    def backward_rv(
        self,
        rv_futu,
        rv_past,
        start,
        stop=None,
        step=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        raise NotImplementedError


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

    def __init__(self, driftmat: np.ndarray, forcevec: np.ndarray, dispmat: np.ndarray):
        _check_initial_state_dimensions(driftmat, forcevec, dispmat)
        dimension = len(driftmat)
        super().__init__(
            (lambda t: driftmat),
            (lambda t: forcevec),
            (lambda t: dispmat),
            dimension=dimension,
        )
        self.driftmat = driftmat
        self.forcevec = forcevec
        self.dispmat = dispmat

    def forward_realization(
        self,
        real,
        start,
        stop,
        _diffusion=1.0,
        **kwargs,
    ):

        if not isinstance(real, np.ndarray):
            raise TypeError(f"Numpy array expected, {type(real)} received.")
        discretised_model = self.discretise(step=stop - start)
        return discretised_model.forward_realization(real, start, _diffusion=_diffusion)

    def forward_rv(
        self,
        rv,
        start,
        stop,
        _diffusion=1.0,
        **kwargs,
    ):

        if not isinstance(rv, pnrv.Normal):
            errormsg = (
                "Closed form transitions in LTI SDE models is only "
                "available for Gaussian initial conditions."
            )
            raise TypeError(errormsg)
        discretised_model = self.discretise(step=stop - start)
        return discretised_model.forward_rv(rv, start, _diffusion=_diffusion)

    def backward_realization(
        self,
        real,
        rv_past,
        start,
        stop=None,
        step=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        raise NotImplementedError

    def backward_rv(
        self,
        rv_futu,
        rv_past,
        start,
        stop=None,
        step=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        raise NotImplementedError

    def discretise(self, step):
        """Returns a discrete transition model (i.e. mild solution to SDE) using matrix
        fraction decomposition.

        That is, matrices A(h) and Q(h) and vector s(h) such
        that the transition is

        .. math:: x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h)) ,

        which is the transition of the mild solution to the LTI SDE.
        """
        if np.linalg.norm(self.forcevec) > 0:
            raise NotImplementedError("MFD does not work for force>0 (yet).")
        ah, qh, _ = matrix_fraction_decomposition(self.driftmat, self.dispmat, step)
        sh = np.zeros(len(ah))
        return discrete_transition.DiscreteLTIGaussian(ah, sh, qh)


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
