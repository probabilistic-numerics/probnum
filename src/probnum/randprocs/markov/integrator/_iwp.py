"""Integrated Brownian motion."""

try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import warnings

import numpy as np
import scipy.special

from probnum import config, linops, randvars
from probnum.randprocs.markov import _markov_process, continuous, discrete
from probnum.randprocs.markov.integrator import _integrator, _preconditioner


class IntegratedWienerProcess(_markov_process.MarkovProcess):
    r"""Integrated Wiener process.

    Convenience access to :math:`\nu` times integrated (:math:`d` dimensional) Wiener processes.

    Parameters
    ----------
    initarg
        Initial time point.
    num_derivatives
        Number of modelled derivatives of the integrated process (''order'', ''number of integrations'').
        Optional. Default is :math:`\nu=1`.
    wiener_process_dimension
        Dimension of the underlying Wiener process.
        Optional. Default is :math:`d=1`.
        The dimension of the integrated Wiener process itself is :math:`d(\nu + 1)`.
    initrv
        Law of the integrated Wiener process at the initial time point.
        Optional. Default is a :math:`d(\nu + 1)` dimensional standard-normal distribution.
    diffuse
        Whether to instantiate a diffuse prior. A diffuse prior has large initial variances.
        Optional. Default is `False`.
        If `True`, and if an initial random variable is not passed, an initial random variable is created,
        where the initial covariance is of the form :math:`\kappa I_{d(\nu + 1)}`
        with :math:`\kappa=10^6`.
        Diffuse priors are used when initial distributions are not known.
        They are common for filtering-based probabilistic ODE solvers.
    forward_implementation
        Implementation of the forward-propagation in the underlying transitions.
        Optional. Default is `classic`. `sqrt` implementation is more computationally expensive, but also more stable.
    backward_implementation
        Implementation of the backward-conditioning in the underlying transitions.
        Optional. Default is `classic`. `sqrt` implementation is more computationally expensive, but also more stable.

    Raises
    ------
    Warning
        If `initrv` is not None and `diffuse` is True.

    Examples
    --------
    >>> iwp1 = IntegratedWienerProcess(initarg=0.)
    >>> print(iwp1)
    <IntegratedWienerProcess with input_dim=1, output_dim=2, dtype=float64>

    >>> iwp2 = IntegratedWienerProcess(initarg=0., num_derivatives=2)
    >>> print(iwp2)
    <IntegratedWienerProcess with input_dim=1, output_dim=3, dtype=float64>

    >>> iwp3 = IntegratedWienerProcess(initarg=0., wiener_process_dimension=10)
    >>> print(iwp3)
    <IntegratedWienerProcess with input_dim=1, output_dim=20, dtype=float64>

    >>> iwp4 = IntegratedWienerProcess(initarg=0., num_derivatives=4, wiener_process_dimension=1)
    >>> print(iwp4)
    <IntegratedWienerProcess with input_dim=1, output_dim=5, dtype=float64>
    """

    def __init__(
        self,
        initarg,
        num_derivatives=1,
        wiener_process_dimension=1,
        initrv=None,
        diffuse=False,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        iwp_transition = IntegratedWienerTransition(
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
        if initrv is not None and diffuse:
            warnings.warn(
                "Parameter `diffuse` has no effect, because an `initrv` has been provided."
            )
        if initrv is None:
            if diffuse:
                scale_cholesky = 1e3
            else:
                scale_cholesky = 1.0
            zeros = np.zeros(iwp_transition.dimension)
            cov_cholesky = scale_cholesky * np.eye(iwp_transition.dimension)
            initrv = randvars.Normal(
                mean=zeros, cov=cov_cholesky ** 2, cov_cholesky=cov_cholesky
            )

        super().__init__(transition=iwp_transition, initrv=initrv, initarg=initarg)


class IntegratedWienerTransition(_integrator.IntegratorTransition, continuous.LTISDE):
    """Integrated Brownian motion in :math:`d` dimensions."""

    def __init__(
        self,
        num_derivatives,
        wiener_process_dimension,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        # initialise BOTH superclasses' inits.
        # I don't like it either, but it does the job.
        _integrator.IntegratorTransition.__init__(
            self,
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
        )
        continuous.LTISDE.__init__(
            self,
            driftmat=self._driftmat,
            forcevec=self._forcevec,
            dispmat=self._dispmat,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

    @cached_property
    def _driftmat(self):
        driftmat_1d = np.diag(np.ones(self.num_derivatives), 1)
        if config.lazy_linalg:
            return linops.Kronecker(
                A=linops.Identity(self.wiener_process_dimension),
                B=linops.Matrix(A=driftmat_1d),
            )
        return np.kron(np.eye(self.wiener_process_dimension), driftmat_1d)

    @cached_property
    def _forcevec(self):
        return np.zeros((self.wiener_process_dimension * (self.num_derivatives + 1)))

    @cached_property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.num_derivatives + 1)
        dispmat_1d[-1] = 1.0  # Unit diffusion

        if config.lazy_linalg:
            return linops.Kronecker(
                A=linops.Identity(self.wiener_process_dimension),
                B=linops.Matrix(A=dispmat_1d.reshape(-1, 1)),
            )
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
            scipy.linalg.pascal(self.num_derivatives + 1, kind="lower", exact=False)
        )
        if config.lazy_linalg:
            state_transition = linops.Kronecker(
                A=linops.Identity(self.wiener_process_dimension),
                B=linops.aslinop(state_transition_1d),
            )
        else:
            state_transition = np.kron(
                np.eye(self.wiener_process_dimension), state_transition_1d
            )
        process_noise_1d = np.flip(scipy.linalg.hilbert(self.num_derivatives + 1))
        if config.lazy_linalg:
            process_noise = linops.Kronecker(
                A=linops.Identity(self.wiener_process_dimension),
                B=linops.aslinop(process_noise_1d),
            )
        else:
            process_noise = np.kron(
                np.eye(self.wiener_process_dimension), process_noise_1d
            )
        empty_shift = np.zeros(
            self.wiener_process_dimension * (self.num_derivatives + 1)
        )

        process_noise_cholesky_1d = np.linalg.cholesky(process_noise_1d)
        if config.lazy_linalg:
            process_noise_cholesky = linops.Kronecker(
                A=linops.Identity(self.wiener_process_dimension),
                B=linops.aslinop(process_noise_cholesky_1d),
            )
        else:
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

        rv = _preconditioner.apply_precon(self.precon.inverse(dt), rv)
        rv, info = self.equivalent_discretisation_preconditioned.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

        info["crosscov"] = self.precon(dt) @ info["crosscov"] @ self.precon(dt).T
        if "gain" in info:
            info["gain"] = self.precon(dt) @ info["gain"] @ self.precon.inverse(dt).T

        return _preconditioner.apply_precon(self.precon(dt), rv), info

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

        rv_obtained = _preconditioner.apply_precon(self.precon.inverse(dt), rv_obtained)
        rv = _preconditioner.apply_precon(self.precon.inverse(dt), rv)
        rv_forwarded = (
            _preconditioner.apply_precon(self.precon.inverse(dt), rv_forwarded)
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

        return _preconditioner.apply_precon(self.precon(dt), rv), info

    def discretise(self, dt):
        """Equivalent discretisation of the process.

        Overwrites matrix-fraction decomposition in the super-class. Only present for
        user's convenience and to maintain a clean interface. Not used for forward_rv,
        etc..
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
        zero_shift = np.zeros(state_trans_mat.shape[0])

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
