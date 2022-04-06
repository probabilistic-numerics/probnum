"""SDE models as transitions."""

from typing import Callable, Optional

import numpy as np

from probnum.randprocs.markov import _transition
from probnum.typing import FloatLike, IntLike


class SDE(_transition.Transition):
    r"""Stochastic differential equation.

    .. math:: d x(t) = g(t, x(t)) d t + l(t, x(t)) d w(t),

    driven by a Wiener process :math:`w` with isotropic diffusion
    :math:`\Gamma(t) = \gamma(t) I_d`.
    """

    def __init__(
        self,
        state_dimension: IntLike,
        wiener_process_dimension: IntLike,
        drift_function: Callable[[FloatLike, np.ndarray], np.ndarray],
        dispersion_function: Callable[[FloatLike, np.ndarray], np.ndarray],
        drift_jacobian: Optional[Callable[[FloatLike, np.ndarray], np.ndarray]],
    ):
        super().__init__(input_dim=state_dimension, output_dim=state_dimension)

        # Mandatory arguments
        self._state_dimension = state_dimension
        self._wiener_process_dimension = wiener_process_dimension
        self._drift_function = drift_function
        self._dispersion_function = dispersion_function

        # Optional arguments
        self._drift_jacobian = drift_jacobian

    @property
    def state_dimension(self):
        return self._state_dimension

    @property
    def wiener_process_dimension(self):
        return self._wiener_process_dimension

    def drift_function(self, t, x):
        return self._drift_function(t, x)

    def dispersion_function(self, t, x):
        return self._dispersion_function(t, x)

    def drift_jacobian(self, t, x):
        if self._drift_jacobian is not None:
            return self._drift_jacobian(t, x)
        raise NotImplementedError("Jacobian not provided.")

    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        return self._forward_realization_via_forward_rv(
            realization,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            **kwargs,
        )

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        raise NotImplementedError("Not available.")

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        **kwargs,
    ):
        return self._backward_realization_via_backward_rv(
            realization_obtained,
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
