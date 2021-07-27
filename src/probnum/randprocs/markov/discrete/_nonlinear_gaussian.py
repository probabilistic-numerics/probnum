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
    def from_ode(
        cls,
        ode,
        prior,
        evlvar=0.0,
    ):

        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ode.f(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(ode.dimension)

        def diff_cholesky(t):
            return np.sqrt(evlvar) * np.eye(ode.dimension)

        def jacobian(t, x):
            return h1 - ode.df(t, h0 @ x) @ h0

        return cls(
            input_dim=prior.dimension,
            output_dim=ode.dimension,
            state_trans_fun=dyna,
            jacob_state_trans_fun=jacobian,
            proc_noise_cov_mat_fun=diff,
            proc_noise_cov_cholesky_fun=diff_cholesky,
        )

    def _duplicate(self, **changes):
        def replace_key(key):
            try:
                return changes[key]
            except KeyError:
                return getattr(self, key)

        input_dim = replace_key("input_dim")
        output_dim = replace_key("output_dim")
        state_trans_fun = replace_key("state_trans_fun")
        proc_noise_cov_mat_fun = replace_key("proc_noise_cov_mat_fun")
        jacob_state_trans_fun = replace_key("jacob_state_trans_fun")
        proc_noise_cov_cholesky_fun = replace_key("proc_noise_cov_cholesky_fun")
        return DiscreteGaussian(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_fun=state_trans_fun,
            proc_noise_cov_mat_fun=proc_noise_cov_mat_fun,
            jacob_state_trans_fun=jacob_state_trans_fun,
            proc_noise_cov_cholesky_fun=proc_noise_cov_cholesky_fun,
        )