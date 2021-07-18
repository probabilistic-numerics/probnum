"""Integrated Ornstein-Uhlenbeck processes."""
import numpy as np

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from probnum.randprocs.markov.continuous import _sde
from probnum.randprocs.markov.continuous.integrator import _integrator, _utils


class IOUP(_integrator.Integrator, _sde.LTISDE):
    """Integrated Ornstein-Uhlenbeck process in :math:`d` dimensions."""

    def __init__(
        self,
        ordint: int,
        spatialdim: int,
        driftspeed: float,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        self.driftspeed = driftspeed

        _integrator.Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        _sde.LTISDE.__init__(
            self,
            driftmat=self._driftmat,
            forcevec=self._forcevec,
            dispmat=self._dispmat,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

    @cached_property
    def _driftmat(self):
        driftmat_1d = np.diag(np.ones(self.ordint), 1)
        driftmat_1d[-1, -1] = -self.driftspeed
        return np.kron(np.eye(self.spatialdim), driftmat_1d)

    @cached_property
    def _forcevec(self):
        force_1d = np.zeros(self.ordint + 1)
        return np.kron(np.ones(self.spatialdim), force_1d)

    @cached_property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = 1.0  # Unit Diffusion
        return np.kron(np.eye(self.spatialdim), dispmat_1d).T

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Fetch things into preconditioned space
        rv = _utils.apply_precon(self.precon.inverse(dt), rv)

        # Apply preconditioning to system matrices
        self.driftmat = self.precon.inverse(dt) @ self.driftmat @ self.precon(dt)
        self.forcevec = self.precon.inverse(dt) @ self.forcevec
        self.dispmat = self.precon.inverse(dt) @ self.dispmat

        # Discretise and propagate
        discretised_model = self.discretise(dt=dt)
        rv, info = discretised_model.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

        # Undo preconditioning and return
        rv = _utils.apply_precon(self.precon(dt), rv)
        info["crosscov"] = self.precon(dt) @ info["crosscov"] @ self.precon(dt).T
        if "gain" in info:
            info["gain"] = self.precon(dt) @ info["gain"] @ self.precon.inverse(dt).T

        self.driftmat = self.precon(dt) @ self.driftmat @ self.precon.inverse(dt)
        self.forcevec = self.precon(dt) @ self.forcevec
        self.dispmat = self.precon(dt) @ self.dispmat

        return rv, info

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
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Fetch things into preconditioned space
        rv_obtained = _utils.apply_precon(self.precon.inverse(dt), rv_obtained)
        rv = _utils.apply_precon(self.precon.inverse(dt), rv)
        rv_forwarded = (
            _utils.apply_precon(self.precon.inverse(dt), rv_forwarded)
            if rv_forwarded is not None
            else None
        )
        gain = (
            self.precon.inverse(dt) @ gain @ self.precon.inverse(dt).T
            if gain is not None
            else None
        )

        # Apply preconditioning to system matrices
        self.driftmat = self.precon.inverse(dt) @ self.driftmat @ self.precon(dt)
        self.forcevec = self.precon.inverse(dt) @ self.forcevec
        self.dispmat = self.precon.inverse(dt) @ self.dispmat

        # Discretise and propagate
        discretised_model = self.discretise(dt=dt)
        rv, info = discretised_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

        # Undo preconditioning and return
        rv = _utils.apply_precon(self.precon(dt), rv)
        self.driftmat = self.precon(dt) @ self.driftmat @ self.precon.inverse(dt)
        self.forcevec = self.precon(dt) @ self.forcevec
        self.dispmat = self.precon(dt) @ self.dispmat
        return rv, info
