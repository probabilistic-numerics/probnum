"""SDE models as transitions."""
import functools
from typing import Callable, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import discrete_transition, transition
from .sde_utils import matrix_fraction_decomposition


class SDE(transition.Transition):
    """Stochastic differential equation.

    .. math:: d x(t) = g(t, x(t)) d t + L(t) d w(t),

    driven by a Wiener process with unit diffusion.
    """

    def __init__(
        self,
        dimension,
        driftfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        dispmatfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        jacobfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
    ):
        self.dimension = dimension
        self.driftfun = driftfun
        self.dispmatfun = dispmatfun
        self.jacobfun = jacobfun
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
        return self._forward_realization_via_forward_rv(
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
        raise NotImplementedError("Not available.")

    def backward_realization(
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
        return self._backward_realization_via_backward_rv(
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
        raise NotImplementedError("Not available.")


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
        dimension,
        driftmatfun: Callable[[FloatArgType], np.ndarray],
        forcevecfun: Callable[[FloatArgType], np.ndarray],
        dispmatfun: Callable[[FloatArgType], np.ndarray],
        mde_atol=1e-5,
        mde_rtol=1e-5,
        mde_solver="LSODA",
    ):

        # Once different filtering and smoothing algorithms are available,
        # replicate the scheme from DiscreteGaussian here, in which
        # the initialisation decides between, e.g., classic and sqrt implementations.

        self.driftmatfun = driftmatfun
        self.forcevecfun = forcevecfun
        super().__init__(
            dimension=dimension,
            driftfun=(lambda t, x: self.driftmatfun(t) @ x + self.forcevecfun(t)),
            dispmatfun=dispmatfun,
            jacobfun=(lambda t, x: self.driftmatfun(t)),
        )

        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        _compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):

        return self._solve_mde_forward(rv, t, dt, _diffusion=_diffusion)

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

    # Forward (and soon, backward) implementation(s)

    def _solve_mde_forward(self, rv, t, dt, _diffusion=1.0):
        """Solve forward moment differential equations (MDEs)."""
        mde, y0 = self._setup_vectorized_mde_forward(
            rv,
            _diffusion=_diffusion,
        )
        sol = scipy.integrate.solve_ivp(
            mde,
            (t, t + dt),
            y0,
            method=self.mde_solver,
            atol=self.mde_atol,
            rtol=self.mde_rtol,
        )
        y_end = sol.y[:, -1]
        new_mean = y_end[: len(rv.mean)]
        new_cov = y_end[len(rv.mean) :].reshape((len(rv.mean), len(rv.mean)))

        # Will come in useful for backward transitions
        # Aka continuous time smoothing.
        sol_mean = lambda t: sol.sol(t)[: len(rv.mean)]
        sol_cov = lambda t: sol.sol(t)[len(rv.mean) :].reshape(
            (len(rv.mean), len(rv.mean))
        )

        return pnrv.Normal(mean=new_mean, cov=new_cov), {
            "sol": sol,
            "sol_mean": sol_mean,
            "sol_cov": sol_cov,
        }

    def _setup_vectorized_mde_forward(self, initrv, _diffusion=1.0):
        """Set up forward moment differential equations (MDEs).

        Compute an ODE vector field that represents the MDEs and is
        compatible with scipy.solve_ivp.
        """
        dim = len(initrv.mean)

        def f(t, y):

            # Undo vectorization
            mean, cov_flat = y[:dim], y[dim:]
            cov = cov_flat.reshape((dim, dim))

            # Apply iteration
            F = self.driftmatfun(t)
            u = self.forcevecfun(t)
            L = self.dispmatfun(t)
            new_mean = F @ mean + u
            new_cov = F @ cov + cov @ F.T + _diffusion * L @ L.T

            # Vectorize outcome
            new_cov_flat = new_cov.flatten()
            y_new = np.hstack((new_mean, new_cov_flat))
            return y_new

        initcov_flat = initrv.cov.flatten()
        y0 = np.hstack((initrv.mean, initcov_flat))

        return f, y0


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
        dimension = len(driftmat)
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
            zeros = np.zeros((self.dimension, self.dimension))
            eye = np.eye(self.dimension)
            driftmat = np.block([[self.driftmat, eye], [zeros, zeros]])
            dispmat = np.block([[self.dispmat], [np.zeros(self.dispmat.shape)]])
            ah_stack, qh_stack, _ = matrix_fraction_decomposition(driftmat, dispmat, dt)
            proj = np.eye(self.dimension, 2 * self.dimension)
            proj_rev = np.flip(proj, axis=1)
            ah = proj @ ah_stack @ proj.T
            sh = proj @ ah_stack @ proj_rev.T @ self.forcevec
            qh = proj @ qh_stack @ proj.T
        else:
            ah, qh, _ = matrix_fraction_decomposition(self.driftmat, self.dispmat, dt)
            sh = np.zeros(len(ah))
        return discrete_transition.DiscreteLTIGaussian(
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
