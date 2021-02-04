"""Discrete transitions."""
from typing import Callable, Optional

import numpy as np

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import transition as trans


class DiscreteGaussian(trans.Transition):
    """Random variable transitions with additive Gaussian noise.

    .. math:: x_{i+1} \\sim \\mathcal{N}(g(t_i, x_i), S(t_i))

    for some (potentially non-linear) dynamics :math:`g` and diffusion matrix :math:`S`.
    This is used for, but not restricted to time-series.

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
    ):
        self.state_trans_fun = state_trans_fun
        self.proc_noise_cov_mat_fun = proc_noise_cov_mat_fun

        def if_no_jacobian(t, x):
            raise NotImplementedError

        self.jacob_state_trans_fun = (
            jacob_state_trans_fun
            if jacob_state_trans_fun is not None
            else if_no_jacobian
        )
        super().__init__()

    def transition_realization(self, real, start, _diffusion=1.0, **kwargs):

        newmean = self.state_trans_fun(start, real)
        newcov = _diffusion * self.proc_noise_cov_mat_fun(start)

        # This is not correct?
        crosscov = np.zeros(newcov.shape)
        return pnrv.Normal(newmean, newcov), {"crosscov": crosscov}

    def transition_rv(self, rv, start, **kwargs):

        raise NotImplementedError


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
    ):

        self.state_trans_mat_fun = state_trans_mat_fun
        self.shift_vec_fun = shift_vec_fun

        super().__init__(
            state_trans_fun=lambda t, x: (
                self.state_trans_mat_fun(t) @ x + self.shift_vec_fun(t)
            ),
            proc_noise_cov_mat_fun=proc_noise_cov_mat_fun,
            jacob_state_trans_fun=lambda t, x: state_trans_mat_fun(t),
        )

    def transition_rv(self, rv, start, _diffusion=1.0, **kwargs):

        if not isinstance(rv, pnrv.Normal):
            raise TypeError(f"Normal RV expected, but {type(rv)} received.")

        state_trans_mat = self.state_trans_mat_fun(start)
        diffmat = _diffusion * self.proc_noise_cov_mat_fun(start)
        shift = self.shift_vec_fun(start)

        new_mean = state_trans_mat @ rv.mean + shift
        new_crosscov = rv.cov @ state_trans_mat.T
        new_cov = state_trans_mat @ new_crosscov + diffmat
        return pnrv.Normal(mean=new_mean, cov=new_cov), {"crosscov": new_crosscov}

    @property
    def dimension(self):
        # risky to evaluate at zero, but works well
        # remove this -- the dimension of discrete transitions is not clear!
        # input dim != output dim is possible...
        # See issue #266
        return len(self.state_trans_mat_fun(0.0).T)


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
    ):
        _check_dimensions(state_trans_mat, shift_vec, proc_noise_cov_mat)

        super().__init__(
            lambda t: state_trans_mat,
            lambda t: shift_vec,
            lambda t: proc_noise_cov_mat,
        )

        self.state_trans_mat = state_trans_mat
        self.shift_vec = shift_vec
        self.proc_noise_cov_mat = proc_noise_cov_mat


def _check_dimensions(state_trans_mat, shift_vec, proc_noise_cov_mat):
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
