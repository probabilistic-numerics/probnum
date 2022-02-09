"""Discrete transitions."""

from typing import Callable, Optional

import numpy as np

from probnum import randvars
from probnum.randprocs.markov import _transition
from probnum.randprocs.markov.discrete import _condition_state
from probnum.typing import ArrayLike, FloatLike, IntLike


class NonlinearGaussian(_transition.Transition):
    r"""Discrete transitions with additive Gaussian noise.

    .. math:: y \sim g(t, x) + v(t), \quad v \sim \mathcal{N}(m(t), S(t))

    for transition function :math:`g: \mathbb{R}^m \rightarrow \mathbb{R}^n`
    and Noise :math:`v`.

    Parameters
    ----------
    input_dim
        Input dimension.
    output_dim
        Output dimension.
    transition_fun
        Transition function :math:`g(t, x)`.
    noise_fun
        Noise :math:`v(t)`.
    transition_fun_jacobian
        Jacobian of the transition function :math:`g(t, x)`.
    """

    def __init__(
        self,
        *,
        input_dim: IntLike,
        output_dim: IntLike,
        transition_fun: Callable[[FloatLike, ArrayLike], ArrayLike],
        noise_fun: Callable[[FloatLike], randvars.RandomVariable],
        transition_fun_jacobian: Optional[
            Callable[[FloatLike, ArrayLike], ArrayLike]
        ] = None,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self._transition_fun = transition_fun
        self._transition_fun_jacobian = transition_fun_jacobian
        self._noise_fun = noise_fun

    @property
    def transition_fun(self):
        return self._transition_fun

    @property
    def transition_fun_jacobian(self):
        if self._transition_fun_jacobian is None:
            raise NotImplementedError
        return self._transition_fun_jacobian

    @property
    def noise_fun(self):
        return self._noise_fun

    def forward_realization(
        self, realization, t, compute_gain=False, _diffusion=1.0, **kwargs
    ):
        fun_eval = self.transition_fun(t, realization)
        noise = _diffusion * self.noise_fun(t)
        return fun_eval + noise, {}

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
    # by LinearNonlinearGaussian and e.g. DiscreteUKFComponent.

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

    @classmethod
    def from_callable(
        cls,
        input_dim: IntLike,
        output_dim: IntLike,
        transition_fun: Callable[[FloatLike, ArrayLike], ArrayLike],
        transition_fun_jacobian: Callable[[FloatLike, ArrayLike], ArrayLike],
    ):
        """Turn a callable into a deterministic transition."""

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            transition_fun=transition_fun,
            transition_fun_jacobian=transition_fun_jacobian,
            noise_fun=lambda t: randvars.Constant(np.zeros(output_dim)),
        )
