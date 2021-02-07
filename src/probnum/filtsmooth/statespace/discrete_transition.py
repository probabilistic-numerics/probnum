"""Discrete transitions."""
from typing import Callable, Optional

import numpy as np

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import transition as trans
from .discrete_transition_utils import (
    backward_rv_classic,
    backward_rv_sqrt,
    forward_rv_classic,
    forward_rv_sqrt,
)

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property, lru_cache
except ImportError:
    from functools import lru_cache

    from cached_property import cached_property


class DiscreteGaussian(trans.Transition):
    """Discrete transitions with additive Gaussian noise.

    .. math:: x_{i+1} \\sim \\mathcal{N}(g(t_i, x_i), S(t_i))

    for some (potentially non-linear) dynamics :math:`g` and process noise covariance matrix :math:`S`.

    Parameters
    ----------
    state_trans_fun :
        State transition function :math:`g=g(t, x)`. Signature: ``state_trans_fun(t, x)``.
    proc_noise_cov_mat_fun :
        Process noise covariance matrix function :math:`S=S(t)`. Signature: ``proc_noise_cov_mat_fun(t)``.
    jacob_state_trans_fun :
        Jacobian of the state transition function :math:`g`, :math:`Jg=Jg(t, x)`.
        Signature: ``jacob_state_trans_fun(t, x)``.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self,
        state_trans_fun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        proc_noise_cov_mat_fun: Callable[[FloatArgType], np.ndarray],
        jacob_state_trans_fun: Optional[
            Callable[[FloatArgType, np.ndarray], np.ndarray]
        ] = None,
        input_dim=None,
        output_dim=None,
    ):
        self.state_trans_fun = state_trans_fun
        self.proc_noise_cov_mat_fun = proc_noise_cov_mat_fun

        def dummy_if_no_jacobian(t, x):
            raise NotImplementedError

        self.jacob_state_trans_fun = (
            jacob_state_trans_fun
            if jacob_state_trans_fun is not None
            else dummy_if_no_jacobian
        )
        super().__init__(input_dim=input_dim, output_dim=output_dim)

    def forward_realization(
        self, real, t, _compute_gain=False, _diffusion=1.0, _linearise_at=None, **kwargs
    ):

        newmean = self.state_trans_fun(start, real)
        newcov = _diffusion * self.proc_noise_cov_mat_fun(start)

        return pnrv.Normal(newmean, newcov), {}

    def forward_rv(
        self, rv, t, _compute_gain=False, _diffusion=1.0, _linearise_at=None, **kwargs
    ):
        # _linearise_at is here, bc DiscreteEKFCOmponent shall inherit from this one?!

        raise NotImplementedError("Not available")

    def backward_realization(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        _linearise_at=None,
        **kwargs,
    ):
        raise NotImplementedError("Not available")

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        _linearise_at=None,
        **kwargs,
    ):
        raise NotImplementedError("Not available")

    @lru_cache(maxsize=None)
    def proc_noise_cov_cholesky_fun(self, t):
        return np.linalg.cholesky(self.proc_noise_cov_mat_fun(t))


class DiscreteLinearGaussian(DiscreteGaussian):
    """Discrete, linear Gaussian transition models of the form.

    .. math:: x_{i+1} \\sim \\mathcal{N}(G(t_i) x_i + v(t_i), S(t_i))

    for some dynamics matrix :math:`G=G(t)`, force vector :math:`v=v(t)`,
    and diffusion matrix :math:`S=S(t)`.


    Parameters
    ----------
    state_trans_mat_fun : callable
        State transition matrix function :math:`G=G(t)`. Signature: ``state_trans_mat_fun(t)``.
    shift_vec_fun : callable
        Shift vector function :math:`v=v(t)`. Signature: ``shift_vec_fun(t)``.
    proc_noise_cov_mat_fun : callable
        Process noise covariance matrix function :math:`S=S(t)`. Signature: ``proc_noise_cov_mat_fun(t)``.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self,
        state_trans_mat_fun: Callable[[FloatArgType], np.ndarray],
        shift_vec_fun: Callable[[FloatArgType], np.ndarray],
        proc_noise_cov_mat_fun: Callable[[FloatArgType], np.ndarray],
        input_dim=None,
        output_dim=None,
        use_forward_rv=forward_rv_classic,
        use_backward_rv=backward_rv_classic,
    ):

        self.state_trans_mat_fun = state_trans_mat_fun
        self.shift_vec_fun = shift_vec_fun

        super().__init__(
            state_trans_fun=lambda t, x: (
                self.state_trans_mat_fun(t) @ x + self.shift_vec_fun(t)
            ),
            proc_noise_cov_mat_fun=proc_noise_cov_mat_fun,
            jacob_state_trans_fun=lambda t, x: state_trans_mat_fun(t),
            input_dim=input_dim,
            output_dim=output_dim,
        )

        self._forward_rv_impl = use_forward_rv
        self._backward_rv_impl = use_backward_rv

    def forward_rv(self, rv, t, _compute_gain=False, _diffusion=1.0, **kwargs):

        return self._forward_rv_impl(
            discrete_transition=self,
            rv=rv,
            t=t,
            compute_gain=_compute_gain,
            _diffusion=_diffusion,
        )

    def forward_realization(self, real, t, _diffusion=1.0, **kwargs):

        zero_cov = np.zeros((len(real), len(real)))
        real_as_rv = pnrv.Normal(mean=real, cov=zero_cov, cov_cholesky=zero_cov)

        return self.forward_rv(
            rv=real_as_rv, t=t, _compute_gain=False, _diffusion=_diffusion
        )

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        **kwargs,
    ):
        return self._backward_rv_impl(
            attained_rv=rv_obtained,
            rv=rv_past,
            forwarded_rv=rv_forwarded,
            gain=gain,
            discrete_transition=self,
            t=t,
            _diffusion=_diffusion,
        )

    def backward_realization(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        **kwargs,
    ):

        zero_cov = np.zeros((len(real), len(real)))
        real_as_rv = pnrv.Normal(mean=real, cov=zero_cov, cov_cholesky=zero_cov)

        return self.backward_rv(
            rv_obtained=real_as_rv,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )


class DiscreteLTIGaussian(DiscreteLinearGaussian):
    """Discrete, linear, time-invariant Gaussian transition models of the form.

    .. math:: x_{i+1} \\sim \\mathcal{N}(G x_i + v, S)

    for some dynamics matrix :math:`G`, force vector :math:`v`,
    and diffusion matrix :math:`S`.

    Parameters
    ----------
    state_trans_mat :
        State transition matrix :math:`G`.
    shift_vec :
        Shift vector :math:`v`.
    proc_noise_cov_mat :
        Process noise covariance matrix :math:`S`.

    Raises
    ------
    TypeError
        If state_trans_mat, shift_vec and proc_noise_cov_mat have incompatible shapes.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self,
        state_trans_mat: np.ndarray,
        shift_vec: np.ndarray,
        proc_noise_cov_mat: np.ndarray,
        input_dim=None,
        output_dim=None,
        use_forward_rv=forward_rv_classic,
        use_backward_rv=backward_rv_classic,
    ):
        _check_dimensions(state_trans_mat, shift_vec, proc_noise_cov_mat)

        super().__init__(
            lambda t: state_trans_mat,
            lambda t: shift_vec,
            lambda t: proc_noise_cov_mat,
            input_dim=input_dim,
            output_dim=output_dim,
            use_forward_rv=use_forward_rv,
            use_backward_rv=use_backward_rv,
        )

        self.state_trans_mat = state_trans_mat
        self.shift_vec = shift_vec
        self.proc_noise_cov_mat = proc_noise_cov_mat

    def proc_noise_cov_cholesky_fun(self, t):
        return self.proc_noise_cov_cholesky

    @cached_property
    def proc_noise_cov_cholesky(self):
        return np.linalg.cholesky(self.proc_noise_cov_mat)


def _check_dimensions(state_trans_mat, shift_vec, proc_noise_cov_mat):
    """LTI SDE model needs matrices which are compatible with each other in size."""
    if state_trans_mat.ndim != 2:
        raise TypeError(
            f"dynamat.ndim=2 expected. dynamat.ndim={state_trans_mat.ndim} received."
        )
    if shift_vec.ndim != 1:
        raise TypeError(
            f"shift_vec.ndim=1 expected. shift_vec.ndim={shift_vec.ndim} received."
        )
    if proc_noise_cov_mat.ndim != 2:
        raise TypeError(
            f"proc_noise_cov_mat.ndim=2 expected. proc_noise_cov_mat.ndim={proc_noise_cov_mat.ndim} received."
        )
    if (
        state_trans_mat.shape[0] != shift_vec.shape[0]
        or shift_vec.shape[0] != proc_noise_cov_mat.shape[0]
        or proc_noise_cov_mat.shape[0] != proc_noise_cov_mat.shape[1]
    ):
        raise TypeError(
            f"Dimension of dynamat, forcevec and diffmat do not align. "
            f"Expected: dynamat.shape=(N,*), forcevec.shape=(N,), diffmat.shape=(N, N).     "
            f"Received: dynamat.shape={state_trans_mat.shape}, forcevec.shape={shift_vec.shape}, "
            f"proc_noise_cov_mat.shape={proc_noise_cov_mat.shape}."
        )
