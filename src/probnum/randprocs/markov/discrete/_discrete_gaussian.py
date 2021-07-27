"""Discrete transitions."""
import typing
import warnings
from functools import lru_cache
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.linalg

from probnum import config, linops, randvars
from probnum.randprocs.markov import _transition
from probnum.randprocs.markov.discrete import _condition_state
from probnum.typing import FloatArgType, IntArgType
from probnum.utils.linalg import cholesky_update, tril_to_positive_tril

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property  # pylint: disable=ungrouped-imports
except ImportError:

    from cached_property import cached_property


class DiscreteGaussian(_transition.Transition):
    r"""Discrete transitions with additive Gaussian noise.

    .. math:: x_{i+1} \sim \mathcal{N}(g(t_i, x_i), S(t_i))

    for some (potentially non-linear) dynamics :math:`g: \mathbb{R}^m \rightarrow \mathbb{R}^n` and process noise covariance matrix :math:`S`.

    Parameters
    ----------
    input_dim
        Dimension of the support of :math:`g` (in terms of :math:`x`), i.e. the input dimension.
    output_dim
        Dimension of the image of :math:`g`, i.e. the output dimension.
    state_trans_fun :
        State transition function :math:`g=g(t, x)`. Signature: ``state_trans_fun(t, x)``.
    proc_noise_cov_mat_fun :
        Process noise covariance matrix function :math:`S=S(t)`. Signature: ``proc_noise_cov_mat_fun(t)``.
    jacob_state_trans_fun :
        Jacobian of the state transition function :math:`g` (with respect to :math:`x`), :math:`Jg=Jg(t, x)`.
        Signature: ``jacob_state_trans_fun(t, x)``.
    proc_noise_cov_cholesky_fun :
        Cholesky factor of the process noise covariance matrix function :math:`\sqrt{S}=\sqrt{S}(t)`. Signature: ``proc_noise_cov_cholesky_fun(t)``.


    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self,
        input_dim: IntArgType,
        output_dim: IntArgType,
        state_trans_fun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        proc_noise_cov_mat_fun: Callable[[FloatArgType], np.ndarray],
        jacob_state_trans_fun: Optional[
            Callable[[FloatArgType, np.ndarray], np.ndarray]
        ] = None,
        proc_noise_cov_cholesky_fun: Optional[
            Callable[[FloatArgType], np.ndarray]
        ] = None,
    ):
        self.state_trans_fun = state_trans_fun
        self.proc_noise_cov_mat_fun = proc_noise_cov_mat_fun

        # "Private", bc. if None, overwritten by the property with the same name
        self._proc_noise_cov_cholesky_fun = proc_noise_cov_cholesky_fun

        def dummy_if_no_jacobian(t, x):
            raise NotImplementedError

        self.jacob_state_trans_fun = (
            jacob_state_trans_fun
            if jacob_state_trans_fun is not None
            else dummy_if_no_jacobian
        )
        super().__init__(input_dim=input_dim, output_dim=output_dim)

    def forward_realization(
        self, realization, t, compute_gain=False, _diffusion=1.0, **kwargs
    ):

        newmean = self.state_trans_fun(t, realization)
        newcov = _diffusion * self.proc_noise_cov_mat_fun(t)

        return randvars.Normal(newmean, newcov), {}

    def forward_rv(self, rv, t, compute_gain=False, _diffusion=1.0, **kwargs):
        raise NotImplementedError("Not available")

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
        raise NotImplementedError("Not available")

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

        # Should we use the _backward_rv_classic here?
        # It is only intractable bc. forward_rv is intractable
        # and assuming its forward formula would yield a valid
        # gain, the backward formula would be valid.
        # This is the case for the UKF, for instance.
        raise NotImplementedError("Not available")

    # Implementations that are the same for all sorts of
    # discrete Gaussian transitions, in particular shared
    # by LinearDiscreteGaussian and e.g. DiscreteUKFComponent.

    def _backward_rv_classic(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        _diffusion=None,
        _linearise_at=None,
    ):

        if rv_forwarded is None or gain is None:
            rv_forwarded, info_forwarded = self.forward_rv(
                rv,
                t=t,
                compute_gain=True,
                _diffusion=_diffusion,
                _linearise_at=_linearise_at,
            )
            gain = info_forwarded["gain"]
        info = {"rv_forwarded": rv_forwarded}
        return (
            _condition_state.condition_state_on_rv(rv_obtained, rv_forwarded, rv, gain),
            info,
        )

    @lru_cache(maxsize=None)
    def proc_noise_cov_cholesky_fun(self, t):
        if self._proc_noise_cov_cholesky_fun is not None:
            return self._proc_noise_cov_cholesky_fun(t)
        covmat = self.proc_noise_cov_mat_fun(t)
        return np.linalg.cholesky(covmat)

    @classmethod
    def from_callable(
        cls,
        input_dim: IntArgType,
        output_dim: IntArgType,
        state_trans_fun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        jacob_state_trans_fun: Callable[[FloatArgType, np.ndarray], np.ndarray],
    ):
        """Turn a callable into a deterministic transition."""

        def diff(t):
            return np.zeros((output_dim, output_dim))

        def diff_cholesky(t):
            return np.zeros((output_dim, output_dim))

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_fun=state_trans_fun,
            jacob_state_trans_fun=jacob_state_trans_fun,
            proc_noise_cov_mat_fun=diff,
            proc_noise_cov_cholesky_fun=diff_cholesky,
        )


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
        input_dim: IntArgType,
        output_dim: IntArgType,
        state_trans_mat_fun: Callable[[FloatArgType], np.ndarray],
        shift_vec_fun: Callable[[FloatArgType], np.ndarray],
        proc_noise_cov_mat_fun: Callable[[FloatArgType], np.ndarray],
        proc_noise_cov_cholesky_fun: Optional[
            Callable[[FloatArgType], np.ndarray]
        ] = None,
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

        if config.lazy_linalg and not isinstance(rv.cov, linops.LinearOperator):
            warnings.warn(
                (
                    "`forward_rv()` received np.ndarray as covariance, while "
                    "`config.lazy_linalg` is set to `True`. This might lead "
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

        if config.lazy_linalg and not (
            isinstance(rv.cov, linops.LinearOperator)
            and isinstance(rv_obtained.cov, linops.LinearOperator)
        ):
            warnings.warn(
                (
                    "`backward_rv()` received np.ndarray as covariance, while "
                    "`config.lazy_linalg` is set to `True`. This might lead "
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
    # _backward_rv_classic is inherited from DiscreteGaussian

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
            if config.lazy_linalg:
                gain = (new_cov.T.inv() @ crosscov.T).T
            else:
                gain = scipy.linalg.solve(new_cov.T, crosscov.T, assume_a="sym").T
            info["gain"] = gain
        return randvars.Normal(new_mean, cov=new_cov), info

    def _forward_rv_sqrt(
        self, rv, t, compute_gain=False, _diffusion=1.0
    ) -> Tuple[randvars.RandomVariable, typing.Dict]:

        if config.lazy_linalg:
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

        if config.lazy_linalg:
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
        proc_noise_cov_cholesky: Optional[np.ndarray] = None,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        _check_dimensions(state_trans_mat, shift_vec, proc_noise_cov_mat)
        output_dim, input_dim = state_trans_mat.shape

        super().__init__(
            input_dim,
            output_dim,
            state_trans_mat_fun=lambda t: state_trans_mat,
            shift_vec_fun=lambda t: shift_vec,
            proc_noise_cov_mat_fun=lambda t: proc_noise_cov_mat,
            proc_noise_cov_cholesky_fun=lambda t: proc_noise_cov_cholesky,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        self.state_trans_mat = state_trans_mat
        self.shift_vec = shift_vec
        self.proc_noise_cov_mat = proc_noise_cov_mat
        self._proc_noise_cov_cholesky = proc_noise_cov_cholesky

    def proc_noise_cov_cholesky_fun(self, t):
        return self.proc_noise_cov_cholesky

    @cached_property
    def proc_noise_cov_cholesky(self):
        if self._proc_noise_cov_cholesky is not None:
            return self._proc_noise_cov_cholesky
        return np.linalg.cholesky(self.proc_noise_cov_mat)

    @classmethod
    def from_linop(
        cls,
        state_trans_mat: np.ndarray,
        shift_vec: np.ndarray,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        """Turn a linear operator (or numpy array) into a deterministic transition."""
        # Currently, this is only a numpy array.
        # In the future, once linops are more widely adopted here, this will become a linop.
        zero_matrix = np.zeros((state_trans_mat.shape[0], state_trans_mat.shape[0]))
        if state_trans_mat.ndim != 2:
            raise ValueError
        return cls(
            state_trans_mat=state_trans_mat,
            shift_vec=shift_vec,
            proc_noise_cov_mat=zero_matrix,
            proc_noise_cov_cholesky=zero_matrix,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


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
