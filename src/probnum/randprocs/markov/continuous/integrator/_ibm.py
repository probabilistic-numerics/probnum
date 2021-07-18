"""Integrated Brownian motion."""

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.special

import probnum.typing as pntype
from probnum import randvars
from probnum.randprocs.markov import discrete
from probnum.randprocs.markov.continuous import _sde
from probnum.randprocs.markov.continuous.integrator import (
    _integrator,
    _preconditioner,
    _utils,
)


class IBM(_integrator.Integrator, _sde.LTISDE):
    """Integrated Brownian motion in :math:`d` dimensions."""

    def __init__(
        self,
        ordint,
        spatialdim,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        # initialise BOTH superclasses' inits.
        # I don't like it either, but it does the job.
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
        return np.kron(np.eye(self.spatialdim), driftmat_1d)

    @cached_property
    def _forcevec(self):
        force_1d = np.zeros(self.ordint + 1)
        return np.kron(np.ones(self.spatialdim), force_1d)

    @cached_property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = 1.0  # Unit diffusion
        return np.kron(np.eye(self.spatialdim), dispmat_1d).T

    @cached_property
    def equivalent_discretisation_preconditioned(self):
        """Discretised IN THE PRECONDITIONED SPACE.

        The preconditioned state transition is the flipped Pascal matrix.
        The preconditioned process noise covariance is the flipped Hilbert matrix.
        The shift is always zero.

        Reference: https://arxiv.org/abs/2012.10106
        """

        state_transition_1d = np.flip(
            scipy.linalg.pascal(self.ordint + 1, kind="lower", exact=False)
        )
        state_transition = np.kron(np.eye(self.spatialdim), state_transition_1d)
        process_noise_1d = np.flip(scipy.linalg.hilbert(self.ordint + 1))
        process_noise = np.kron(np.eye(self.spatialdim), process_noise_1d)
        empty_shift = np.zeros(self.spatialdim * (self.ordint + 1))

        process_noise_cholesky_1d = np.linalg.cholesky(process_noise_1d)
        process_noise_cholesky = np.kron(
            np.eye(self.spatialdim), process_noise_cholesky_1d
        )

        return discrete.DiscreteLTIGaussian(
            state_trans_mat=state_transition,
            shift_vec=empty_shift,
            proc_noise_cov_mat=process_noise,
            proc_noise_cov_cholesky=process_noise_cholesky,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
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
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        rv = _utils.apply_precon(self.precon.inverse(dt), rv)
        rv, info = self.equivalent_discretisation_preconditioned.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

        info["crosscov"] = self.precon(dt) @ info["crosscov"] @ self.precon(dt).T
        if "gain" in info:
            info["gain"] = self.precon(dt) @ info["gain"] @ self.precon.inverse(dt).T

        return _utils.apply_precon(self.precon(dt), rv), info

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

        rv, info = self.equivalent_discretisation_preconditioned.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

        return _utils.apply_precon(self.precon(dt), rv), info

    def discretise(self, dt):
        """Equivalent discretisation of the process.

        Overwrites matrix-fraction decomposition in the super-class.
        Only present for user's convenience and to maintain a clean
        interface. Not used for forward_rv, etc..
        """
        state_trans_mat = (
            self.precon(dt)
            @ self.equivalent_discretisation_preconditioned.state_trans_mat
            @ self.precon.inverse(dt)
        )
        proc_noise_cov_mat = (
            self.precon(dt)
            @ self.equivalent_discretisation_preconditioned.proc_noise_cov_mat
            @ self.precon(dt).T
        )
        zero_shift = np.zeros(len(state_trans_mat))

        # The Cholesky factor of the process noise covariance matrix of the IBM
        # always exists, even for non-square root implementations.
        proc_noise_cov_cholesky = (
            self.precon(dt)
            @ self.equivalent_discretisation_preconditioned.proc_noise_cov_cholesky
        )

        return discrete.DiscreteLTIGaussian(
            state_trans_mat=state_trans_mat,
            shift_vec=zero_shift,
            proc_noise_cov_mat=proc_noise_cov_mat,
            proc_noise_cov_cholesky=proc_noise_cov_cholesky,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.forward_implementation,
        )
