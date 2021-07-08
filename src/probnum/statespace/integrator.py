"""Integrated processes such as the integrated Wiener process or the Matern process.

This is the sucessor of the former ODEPrior.
"""
try:
    # cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.sparse
import scipy.special

import probnum.typing as pntype
from probnum import linops, randvars

from . import discrete_transition, sde
from .preconditioner import NordsieckLikeCoordinates


class Integrator:
    """An integrator is a special kind of SDE, where the :math:`i` th coordinate models
    the :math:`i` th derivative."""

    def __init__(self, ordint, spatialdim):
        self.ordint = ordint
        self.spatialdim = spatialdim
        self.precon = NordsieckLikeCoordinates.from_order(self.ordint, self.spatialdim)

    def proj2coord(self, coord: int) -> np.ndarray:
        """Projection matrix to :math:`i` th coordinates.

        Computes the matrix

        .. math:: H_i = \\left[ I_d \\otimes e_i \\right] P^{-1},

        where :math:`e_i` is the :math:`i` th unit vector,
        that projects to the :math:`i` th coordinate of a vector.
        If the ODE is multidimensional, it projects to **each** of the
        :math:`i` th coordinates of each ODE dimension.

        Parameters
        ----------
        coord : int
            Coordinate index :math:`i` which to project to.
            Expected to be in range :math:`0 \\leq i \\leq q + 1`.

        Returns
        -------
        np.ndarray, shape=(d, d*(q+1))
            Projection matrix :math:`H_i`.
        """
        projvec1d = np.eye(self.ordint + 1)[:, coord]
        projmat1d = projvec1d.reshape((1, self.ordint + 1))
        projmat = np.kron(np.eye(self.spatialdim), projmat1d)
        return projmat

    @property
    def _derivwise2coordwise_projmat(self) -> np.ndarray:
        r"""Projection matrix to change the ordering of the state representation in an :class:`Integrator` from coordinate-wise to derivative-wise representation.

        A coordinate-wise ordering is

        .. math:: (y_1, \dot y_1, \ddot y_1, y_2, \dot y_2, ..., y_d^{(\nu)})

        and a derivative-wise ordering is

        .. math:: (y_1, y_2, ..., y_d, \dot y_1, \dot y_2, ..., \dot y_d, \ddot y_1, ..., y_d^{(\nu)}).

        Default representation in an :class:`Integrator` is coordinate-wise ordering, but sometimes, derivative-wise ordering is more convenient.

        See Also
        --------
        :attr:`Integrator._convert_coordwise_to_derivwise`
        :attr:`Integrator._convert_derivwise_to_coordwise`

        """
        dim = (self.ordint + 1) * self.spatialdim
        projmat = np.zeros((dim, dim))
        E = np.eye(dim)
        for q in range(self.ordint + 1):

            projmat[q :: (self.ordint + 1)] = E[
                q * self.spatialdim : (q + 1) * self.spatialdim
            ]
        return projmat

    @property
    def _coordwise2derivwise_projmat(self) -> np.ndarray:
        r"""Projection matrix to change the ordering of the state representation in an :class:`Integrator` from derivative-wise to coordinate-wise representation.

        A coordinate-wise ordering is

        .. math:: (y_1, \dot y_1, \ddot y_1, y_2, \dot y_2, ..., y_d^{(\nu)})

        and a derivative-wise ordering is

        .. math:: (y_1, y_2, ..., y_d, \dot y_1, \dot y_2, ..., \dot y_d, \ddot y_1, ..., y_d^{(\nu)}).

        Default representation in an :class:`Integrator` is coordinate-wise ordering, but sometimes, derivative-wise ordering is more convenient.

        See Also
        --------
        :attr:`Integrator._convert_coordwise_to_derivwise`
        :attr:`Integrator._convert_derivwise_to_coordwise`

        """
        return self._derivwise2coordwise_projmat.T

    @staticmethod
    def _convert_coordwise_to_derivwise(
        state: np.ndarray, ordint: pntype.IntArgType, spatialdim: pntype.IntArgType
    ) -> np.ndarray:
        """Convert coordinate-wise representation to derivative-wise representation.

        Lightweight call to the respective property in :class:`Integrator`.

        Parameters
        ----------
        state:
            State to be converted. Assumed to be in coordinate-wise representation.
        ordint:
            Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
        spatialdim:
            Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.

        See Also
        --------
        :attr:`Integrator._coordwise2derivwise_projmat`
        :attr:`Integrator._derivwise2coordwise_projmat`
        """
        projmat = Integrator(ordint, spatialdim)._coordwise2derivwise_projmat
        return projmat @ state

    @staticmethod
    def _convert_derivwise_to_coordwise(
        state: np.ndarray, ordint: pntype.IntArgType, spatialdim: pntype.IntArgType
    ) -> np.ndarray:
        """Convert coordinate-wise representation to derivative-wise representation.

        Lightweight call to the respective property in :class:`Integrator`.

        Parameters
        ----------
        state:
            State to be converted. Assumed to be in derivative-wise representation.
        ordint:
            Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
        spatialdim:
            Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.


        See Also
        --------
        :attr:`Integrator._coordwise2derivwise_projmat`
        :attr:`Integrator._derivwise2coordwise_projmat`
        """
        projmat = Integrator(ordint, spatialdim)._derivwise2coordwise_projmat
        return projmat @ state


class IBM(Integrator, sde.LTISDE):
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
        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
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
        # driftmat_1d = scipy.sparse.diags(np.ones(self.ordint), offsets=1)
        # return np.kron(np.eye(self.spatialdim), driftmat_1d)
        return linops.Kronecker(
            A=linops.Identity(self.spatialdim), B=linops.Matrix(A=driftmat_1d)
        )

    @cached_property
    def _forcevec(self):
        # force_1d = np.zeros(self.ordint + 1)
        # return np.kron(np.ones(self.spatialdim), force_1d)
        return np.zeros((self.spatialdim * (self.ordint + 1)))

    @cached_property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = 1.0  # Unit diffusion

        # return np.kron(np.eye(self.spatialdim), dispmat_1d).T

        return linops.Kronecker(
            A=linops.Identity(self.spatialdim),
            B=linops.Matrix(A=dispmat_1d.reshape(-1, 1)),
        )

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
        # state_transition = np.kron(np.eye(self.spatialdim), state_transition_1d)
        state_transition = linops.Kronecker(
            A=linops.Identity(self.spatialdim), B=linops.aslinop(state_transition_1d)
        )
        process_noise_1d = np.flip(scipy.linalg.hilbert(self.ordint + 1))
        # process_noise = np.kron(np.eye(self.spatialdim), process_noise_1d)
        process_noise = linops.Kronecker(
            A=linops.Identity(self.spatialdim), B=linops.aslinop(process_noise_1d)
        )
        empty_shift = np.zeros(self.spatialdim * (self.ordint + 1))

        process_noise_cholesky_1d = np.linalg.cholesky(process_noise_1d)
        # process_noise_cholesky = np.kron(
        #     np.eye(self.spatialdim), process_noise_cholesky_1d
        # )
        process_noise_cholesky = linops.Kronecker(
            A=linops.Identity(self.spatialdim),
            B=linops.aslinop(process_noise_cholesky_1d),
        )

        return discrete_transition.DiscreteLTIGaussian(
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

        rv = _apply_precon(self.precon.inverse(dt), rv)
        rv, info = self.equivalent_discretisation_preconditioned.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

        info["crosscov"] = self.precon(dt) @ info["crosscov"] @ self.precon(dt).T
        if "gain" in info:
            info["gain"] = self.precon(dt) @ info["gain"] @ self.precon.inverse(dt).T

        return _apply_precon(self.precon(dt), rv), info

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

        rv_obtained = _apply_precon(self.precon.inverse(dt), rv_obtained)
        rv = _apply_precon(self.precon.inverse(dt), rv)
        rv_forwarded = (
            _apply_precon(self.precon.inverse(dt), rv_forwarded)
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

        return _apply_precon(self.precon(dt), rv), info

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
        zero_shift = np.zeros(state_trans_mat.shape[0])

        # The Cholesky factor of the process noise covariance matrix of the IBM
        # always exists, even for non-square root implementations.
        proc_noise_cov_cholesky = (
            self.precon(dt)
            @ self.equivalent_discretisation_preconditioned.proc_noise_cov_cholesky
        )

        return discrete_transition.DiscreteLTIGaussian(
            state_trans_mat=state_trans_mat,
            shift_vec=zero_shift,
            proc_noise_cov_mat=proc_noise_cov_mat,
            proc_noise_cov_cholesky=proc_noise_cov_cholesky,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.forward_implementation,
        )


class IOUP(Integrator, sde.LTISDE):
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

        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
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
        rv = _apply_precon(self.precon.inverse(dt), rv)

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
        rv = _apply_precon(self.precon(dt), rv)
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
        rv_obtained = _apply_precon(self.precon.inverse(dt), rv_obtained)
        rv = _apply_precon(self.precon.inverse(dt), rv)
        rv_forwarded = (
            _apply_precon(self.precon.inverse(dt), rv_forwarded)
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
        rv = _apply_precon(self.precon(dt), rv)
        self.driftmat = self.precon(dt) @ self.driftmat @ self.precon.inverse(dt)
        self.forcevec = self.precon(dt) @ self.forcevec
        self.dispmat = self.precon(dt) @ self.dispmat
        return rv, info


class Matern(Integrator, sde.LTISDE):
    """Matern process in :math:`d` dimensions."""

    def __init__(
        self,
        ordint: int,
        spatialdim: int,
        lengthscale: float,
        forward_implementation="classic",
        backward_implementation="classic",
    ):

        self.lengthscale = lengthscale

        Integrator.__init__(self, ordint=ordint, spatialdim=spatialdim)
        sde.LTISDE.__init__(
            self,
            driftmat=self._driftmat,
            forcevec=self._forcevec,
            dispmat=self._dispmat,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

    @property
    def _driftmat(self):
        driftmat = np.diag(np.ones(self.ordint), 1)
        nu = self.ordint + 0.5
        D, lam = self.ordint + 1, np.sqrt(2 * nu) / self.lengthscale
        driftmat[-1, :] = np.array(
            [-scipy.special.binom(D, i) * lam ** (D - i) for i in range(D)]
        )
        return np.kron(np.eye(self.spatialdim), driftmat)

    @property
    def _forcevec(self):
        force_1d = np.zeros(self.ordint + 1)
        return np.kron(np.ones(self.spatialdim), force_1d)

    @property
    def _dispmat(self):
        dispmat_1d = np.zeros(self.ordint + 1)
        dispmat_1d[-1] = 1.0  # Unit diffusion
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
        rv = _apply_precon(self.precon.inverse(dt), rv)

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
        rv = _apply_precon(self.precon(dt), rv)
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
        rv_obtained = _apply_precon(self.precon.inverse(dt), rv_obtained)
        rv = _apply_precon(self.precon.inverse(dt), rv)
        rv_forwarded = (
            _apply_precon(self.precon.inverse(dt), rv_forwarded)
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
        rv = _apply_precon(self.precon(dt), rv)
        self.driftmat = self.precon(dt) @ self.driftmat @ self.precon.inverse(dt)
        self.forcevec = self.precon(dt) @ self.forcevec
        self.dispmat = self.precon(dt) @ self.dispmat
        return rv, info


def _apply_precon(precon, rv):

    # There is no way of checking whether `rv` has its Cholesky factor computed already or not.
    # Therefore, since we need to update the Cholesky factor for square-root filtering,
    # we also update the Cholesky factor for non-square-root algorithms here,
    # which implies additional cost.
    # See Issues #319 and #329.
    # When they are resolved, this function here will hopefully be superfluous.

    new_mean = precon @ rv.mean
    new_cov_cholesky = precon @ rv.cov_cholesky  # precon is diagonal, so this is valid
    new_cov = new_cov_cholesky @ new_cov_cholesky.T

    return randvars.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky)
