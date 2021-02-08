"""SDE models as transitions."""
import functools
from typing import Callable, Optional

import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import discrete_transition, transition
from .discrete_transition_utils import backward_rv_classic, forward_rv_classic
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
        self.dimension = dimension
        super().__init__(input_dim=dimension, output_dim=dimension)

    def forward_realization(
        self,
        real,
        t,
        dt=None,
        _compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        return self._forward_realization_as_rv(
            real,
            t=t,
            dt=dt,
            _compute_gain=_compute_gain,
            _diffusion=_diffusion,
            **kwargs,
        )

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        _compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
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
        **kwargs,
    ):
        return self._backward_realization_as_rv(
            real_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
            **kwargs,
        )

    def backward_rv(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        **kwargs,
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
        moment_equation_stepsize=np.nan,  # use this one for forward and backward!
    ):
        self.driftmatfun = driftmatfun
        self.forcevecfun = forcevecfun
        super().__init__(
            driftfun=(lambda t, x: driftmatfun(t) @ x + forcevecfun(t)),
            dispmatfun=dispmatfun,
            jacobfun=(lambda t, x: driftmatfun(t)),
            dimension=dimension,
        )
        assert dimension is not None

        self.moment_equation_stepsize = moment_equation_stepsize

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        _compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):

        return linear_sde_statistics(
            rv,
            t,
            dt,
            self.moment_equation_stepsize,
            self.driftfun,
            self.driftmatfun,
            self.dispmatfun,
            _diffusion=_diffusion,
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
        raise NotImplementedError("Not available (yet).")


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
        use_forward_rv=forward_rv_classic,
        use_backward_rv=backward_rv_classic,
    ):
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

        self.use_forward_rv = use_forward_rv
        self.use_backward_rv = use_backward_rv

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        _compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        discretised_model = self.discretise(dt=dt)
        return discretised_model.forward_rv(rv, t, _diffusion=_diffusion)

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
            raise NotImplementedError("MFD does not work for force>0 (yet).")
        ah, qh, _ = matrix_fraction_decomposition(self.driftmat, self.dispmat, dt)
        sh = np.zeros(len(ah))
        return discrete_transition.DiscreteLTIGaussian(
            ah,
            sh,
            qh,
            use_forward_rv=self.use_forward_rv,
            use_backward_rv=self.use_backward_rv,
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
