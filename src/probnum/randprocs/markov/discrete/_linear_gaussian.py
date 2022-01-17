"""Discrete, linear Gaussian transitions."""
import typing
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.linalg

from probnum import config, linops, randvars
from probnum.randprocs.markov.discrete import _nonlinear_gaussian
from probnum.typing import FloatLike, IntLike
from probnum.utils.linalg import cholesky_update, tril_to_positive_tril


class LinearGaussian(_nonlinear_gaussian.NonlinearGaussian):
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
    :class:`NonlinearGaussianLinearModel`
    """

    def __init__(
        self,
        input_dim: IntLike,
        output_dim: IntLike,
        state_trans_mat_fun: Callable[[FloatLike], np.ndarray],
        shift_vec_fun: Callable[[FloatLike], np.ndarray],
        proc_noise_cov_mat_fun: Callable[[FloatLike], np.ndarray],
        proc_noise_cov_cholesky_fun: Optional[Callable[[FloatLike], np.ndarray]] = None,
        forward_implementation="classic",
        backward_implementation="classic",
    ):

        # Choose implementation for forward and backward transitions
        choose_forward_implementation = {
            "classic": self._forward_rv_classic,
            "sqrt": self._forward_rv_sqrt,
        }
        choose_backward_implementation = {
            "classic": self._backward_rv_classic,
            "sqrt": self._backward_rv_sqrt,
            "joseph": self._backward_rv_joseph,
        }
        self._forward_implementation = choose_forward_implementation[
            forward_implementation
        ]
        self._backward_implementation = choose_backward_implementation[
            backward_implementation
        ]

        self.state_trans_mat_fun = state_trans_mat_fun
        self.shift_vec_fun = shift_vec_fun
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_fun=lambda t, x: (
                self.state_trans_mat_fun(t) @ x + self.shift_vec_fun(t)
            ),
            proc_noise_cov_mat_fun=proc_noise_cov_mat_fun,
            proc_noise_cov_cholesky_fun=proc_noise_cov_cholesky_fun,
            jacob_state_trans_fun=lambda t, x: state_trans_mat_fun(t),
        )

    def forward_rv(self, rv, t, compute_gain=False, _diffusion=1.0, **kwargs):

        if config.matrix_free and not isinstance(rv.cov, linops.LinearOperator):
            warnings.warn(
                (
                    "`forward_rv()` received np.ndarray as covariance, while "
                    "`config.matrix_free` is set to `True`. This might lead "
                    "to unexpected behavior regarding data types."
                ),
                RuntimeWarning,
            )

        return self._forward_implementation(
            rv=rv,
            t=t,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
        )

    def forward_realization(self, realization, t, _diffusion=1.0, **kwargs):

        return self._forward_realization_via_forward_rv(
            realization, t=t, compute_gain=False, _diffusion=_diffusion
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

        if config.matrix_free and not (
            isinstance(rv.cov, linops.LinearOperator)
            and isinstance(rv_obtained.cov, linops.LinearOperator)
        ):
            warnings.warn(
                (
                    "`backward_rv()` received np.ndarray as covariance, while "
                    "`config.matrix_free` is set to `True`. This might lead "
                    "to unexpected behavior regarding data types."
                ),
                RuntimeWarning,
            )

        return self._backward_implementation(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
        **kwargs,
    ):
        return self._backward_realization_via_backward_rv(
            realization_obtained,
            rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

    # Forward and backward implementations
    # _backward_rv_classic is inherited from NonlinearGaussian

    def _forward_rv_classic(
        self, rv, t, compute_gain=False, _diffusion=1.0
    ) -> Tuple[randvars.RandomVariable, typing.Dict]:
        H = self.state_trans_mat_fun(t)
        R = self.proc_noise_cov_mat_fun(t)
        shift = self.shift_vec_fun(t)

        new_mean = H @ rv.mean + shift
        crosscov = rv.cov @ H.T
        new_cov = H @ crosscov + _diffusion * R
        info = {"crosscov": crosscov}
        if compute_gain:
            if config.matrix_free:
                # gain = (new_cov.T.inv() @ crosscov.T).T
                gain = crosscov @ new_cov.inv()
            else:
                gain = scipy.linalg.solve(new_cov.T, crosscov.T, assume_a="sym").T
            info["gain"] = gain
        return randvars.Normal(new_mean, cov=new_cov), info

    def _forward_rv_sqrt(
        self, rv, t, compute_gain=False, _diffusion=1.0
    ) -> Tuple[randvars.RandomVariable, typing.Dict]:

        if config.matrix_free:
            raise NotImplementedError(
                "Sqrt-implementation does not work with linops for now."
            )

        H = self.state_trans_mat_fun(t)
        SR = self.proc_noise_cov_cholesky_fun(t)
        shift = self.shift_vec_fun(t)

        new_mean = H @ rv.mean + shift
        new_cov_cholesky = cholesky_update(
            H @ rv.cov_cholesky, np.sqrt(_diffusion) * SR
        )
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        crosscov = rv.cov @ H.T
        info = {"crosscov": crosscov}
        if compute_gain:
            info["gain"] = scipy.linalg.cho_solve(
                (new_cov_cholesky, True), crosscov.T
            ).T
        return (
            randvars.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky),
            info,
        )

    def _backward_rv_sqrt(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=1.0,
    ) -> Tuple[randvars.RandomVariable, typing.Dict]:
        """See Section 4.1f of:

        ``https://www.sciencedirect.com/science/article/abs/pii/S0005109805001810``.
        """
        # forwarded_rv is ignored in square-root smoothing.

        if config.matrix_free:
            raise NotImplementedError(
                "Sqrt-implementation does not work with linops for now."
            )

        # Smoothing updates need the gain, but
        # filtering updates "compute their own".
        # Thus, if we are doing smoothing (|cov_obtained|>0) an the gain is not provided,
        # make an extra prediction to compute the gain.
        if gain is None:
            if np.linalg.norm(rv_obtained.cov) > 0:
                rv_forwarded, info_forwarded = self.forward_rv(
                    rv, t=t, compute_gain=True, _diffusion=_diffusion
                )
                gain = info_forwarded["gain"]
            else:
                gain = np.zeros((len(rv.mean), len(rv_obtained.mean)))

        state_trans = self.state_trans_mat_fun(t)
        proc_noise_chol = np.sqrt(_diffusion) * self.proc_noise_cov_cholesky_fun(t)
        shift = self.shift_vec_fun(t)

        chol_past = rv.cov_cholesky
        chol_obtained = rv_obtained.cov_cholesky

        output_dim = self.output_dim
        input_dim = self.input_dim

        zeros_bottomleft = np.zeros((output_dim, output_dim))
        zeros_middleright = np.zeros((output_dim, input_dim))

        blockmat = np.block(
            [
                [chol_past.T @ state_trans.T, chol_past.T],
                [proc_noise_chol.T, zeros_middleright],
                [zeros_bottomleft, chol_obtained.T @ gain.T],
            ]
        )
        big_triu = np.linalg.qr(blockmat, mode="r")
        new_chol_triu = big_triu[
            output_dim : (output_dim + input_dim), output_dim : (output_dim + input_dim)
        ]

        # If no initial gain was provided, compute it from the QR-results
        # This is required in the Kalman update, where, other than in the smoothing update,
        # no initial gain was provided.
        # Recall that above, gain was set to zero in this setting.
        if np.linalg.norm(gain) == 0.0:
            R1 = big_triu[:output_dim, :output_dim]
            R12 = big_triu[:output_dim, output_dim:]
            gain = scipy.linalg.solve_triangular(R1, R12, lower=False).T

        new_mean = rv.mean + gain @ (rv_obtained.mean - state_trans @ rv.mean - shift)
        new_cov_cholesky = tril_to_positive_tril(new_chol_triu.T)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T

        info = {"rv_forwarded": rv_forwarded}
        return randvars.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky), info

    def _backward_rv_joseph(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=None,
    ) -> Tuple[randvars.RandomVariable, typing.Dict]:
        # forwarded_rv is ignored in Joseph updates.

        if gain is None:
            rv_forwarded, info_forwarded = self.forward_rv(
                rv, t=t, compute_gain=True, _diffusion=_diffusion
            )
            gain = info_forwarded["gain"]

        H = self.state_trans_mat_fun(t)
        R = _diffusion * self.proc_noise_cov_mat_fun(t)
        shift = self.shift_vec_fun(t)

        new_mean = rv.mean + gain @ (rv_obtained.mean - H @ rv.mean - shift)
        joseph_factor = np.eye(len(rv.mean)) - gain @ H
        new_cov = (
            joseph_factor @ rv.cov @ joseph_factor.T
            + gain @ R @ gain.T
            + gain @ rv_obtained.cov @ gain.T
        )

        info = {"rv_forwarded": rv_forwarded}
        return randvars.Normal(new_mean, new_cov), info
