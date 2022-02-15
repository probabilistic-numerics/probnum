"""Temporary interface.

Will eventually be removed as a part of the refactoring detailed in issue #627.
"""

from typing import Dict, Tuple

from probnum import randprocs, randvars


class _LinearizationInterface:
    """Interface for extended Kalman filtering components."""

    def __init__(
        self,
        non_linear_model,
    ) -> None:

        self.non_linear_model = non_linear_model

    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ) -> Tuple[randvars.Normal, Dict]:
        """Approximate forward-propagation of a realization of a random variable."""
        compute_jacobian_at = (
            _linearise_at
            if _linearise_at is not None
            else randvars.Constant(realization)
        )
        linearized_model = self.linearize(t=t, at_this_rv=compute_jacobian_at)
        return linearized_model.forward_realization(
            realization=realization,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
        )

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ) -> Tuple[randvars.Normal, Dict]:
        """Approximate forward-propagation of a random variable."""

        compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
        linearized_model = self.linearize(t=t, at_this_rv=compute_jacobian_at)
        return linearized_model.forward_rv(
            rv=rv,
            t=t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
        )

    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        """Approximate backward-propagation of a realization of a random variable."""
        return self._backward_realization_via_backward_rv(
            realization_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
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
        _linearise_at=None,
    ):
        """Approximate backward-propagation of a random variable."""

        compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
        linearized_model = self.linearize(t=t, at_this_rv=compute_jacobian_at)
        return linearized_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
        )

    def linearize(
        self, t, at_this_rv: randvars.RandomVariable
    ) -> randprocs.markov.Transition:
        """Linearize the transition and make it tractable."""
        raise NotImplementedError
