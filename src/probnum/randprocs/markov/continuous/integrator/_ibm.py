"""Integrated Brownian motion."""

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.special

from probnum import randvars
from probnum.randprocs.markov import _markov_process, discrete
from probnum.randprocs.markov.continuous import _sde
from probnum.randprocs.markov.continuous.integrator import _integrator, _utils


class IntegratedWienerProcess(_markov_process.MarkovProcess):
    """Convenience access to integrated Wiener processes.

    Examples
    --------
    >>> iwp1 = IntegratedWienerProcess(initarg=0.)
    >>> print(iwp1)
    <IntegratedWienerProcess with input_dim=1, output_dim=2, dtype=float64>

    >>> iwp2 = IntegratedWienerProcess(initarg=0., nu=2)
    >>> print(iwp2)
    <IntegratedWienerProcess with input_dim=1, output_dim=3, dtype=float64>

    >>> iwp3 = IntegratedWienerProcess(initarg=0., wiener_process_dimension=10)
    >>> print(iwp3)
    <IntegratedWienerProcess with input_dim=1, output_dim=20, dtype=float64>

    >>> iwp4 = IntegratedWienerProcess(initarg=0., nu=4, wiener_process_dimension=1)
    >>> print(iwp4)
    <IntegratedWienerProcess with input_dim=1, output_dim=5, dtype=float64>
    """

    def __init__(
        self,
        initarg,
        nu=1,
        wiener_process_dimension=1,
        initrv=None,
        forward_implementation="classic",
        backward_implementation="classic",
    ):

        iwp_transition = IntegratedWienerProcessTransition(
            nu=nu,
            wiener_process_dimension=wiener_process_dimension,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
        if initrv is None:
            zeros = np.zeros(iwp_transition.dimension)
            eye = np.eye(iwp_transition.dimension)
            initrv = randvars.Normal(mean=zeros, cov=eye, cov_cholesky=eye)

        super().__init__(transition=iwp_transition, initrv=initrv, initarg=initarg)


class IntegratedWienerProcessTransition(_integrator.IntegratorTransition, _sde.LTISDE):
    """Integrated Brownian motion in :math:`d` dimensions."""

    def __init__(
        self,
        nu,
        wiener_process_dimension,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        # initialise BOTH superclasses' inits.
        # I don't like it either, but it does the job.
        _integrator.IntegratorTransition.__init__(
            self, nu=nu, wiener_process_dimension=wiener_process_dimension
        )
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
        driftmat_1d = np.diag(np.ones(self.nu), 1)
        return np.kron(np.eye(self.wiener_process_dimension), driftmat_1d)

    @cached_property
    def _forcevec(self):
        force_1d = np.zeros(self.nu + 1)
        return np.kron(np.ones(self.wiener_process_dimension), force_1d)

    @cached_property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.nu + 1)
        dispmat_1d[-1] = 1.0  # Unit diffusion
        return np.kron(np.eye(self.wiener_process_dimension), dispmat_1d).T

    @cached_property
    def equivalent_discretisation_preconditioned(self):
        """Discretised IN THE PRECONDITIONED SPACE.

        The preconditioned state transition is the flipped Pascal matrix.
        The preconditioned process noise covariance is the flipped Hilbert matrix.
        The shift is always zero.

        Reference: https://arxiv.org/abs/2012.10106
        """

        state_transition_1d = np.flip(
            scipy.linalg.pascal(self.nu + 1, kind="lower", exact=False)
        )
        state_transition = np.kron(
            np.eye(self.wiener_process_dimension), state_transition_1d
        )
        process_noise_1d = np.flip(scipy.linalg.hilbert(self.nu + 1))
        process_noise = np.kron(np.eye(self.wiener_process_dimension), process_noise_1d)
        empty_shift = np.zeros(self.wiener_process_dimension * (self.nu + 1))

        process_noise_cholesky_1d = np.linalg.cholesky(process_noise_1d)
        process_noise_cholesky = np.kron(
            np.eye(self.wiener_process_dimension), process_noise_cholesky_1d
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
