"""Gaussian IVP filtering and smoothing."""

from typing import Callable, Optional

import numpy as np

from probnum import filtsmooth, randprocs, randvars, statespace

from ..ode import IVP
from ..odesolver import ODESolver
from .initialize import (
    initialize_odefilter_with_rk,
    initialize_odefilter_with_taylormode,
)
from .kalman_odesolution import KalmanODESolution


class GaussianIVPFilter(ODESolver):
    """ODE solver that uses a Gaussian filter.

    This is based on continuous-discrete Gaussian filtering.

    Note: this is specific for IVPs and does not apply without
    further considerations to, e.g., BVPs.

    Parameters
    ----------
    ivp
        Initial value problem to be solved.
    prior_process
        Prior Gauss-Markov process.
    measurement_model
        ODE measurement model.
    with_smoothing
        To smooth after the solve or not to smooth after the solve.
    init_implementation
        Initialization algorithm. Either via Scipy (``initialize_odefilter_with_rk``) or via Taylor-mode AD (``initialize_odefilter_with_taylormode``).
        For more convenient construction, consider :func:`GaussianIVPFilter.construct_with_rk_init` and :func:`GaussianIVPFilter.construct_with_taylormode_init`.
    """

    def __init__(
        self,
        ivp: IVP,
        prior_process: randprocs.MarkovProcess,
        measurement_model: statespace.DiscreteGaussian,
        with_smoothing: bool,
        init_implementation: Callable[
            [
                Callable,
                np.ndarray,
                float,
                statespace.Integrator,
                randvars.Normal,
                Optional[Callable],
            ],
            randvars.Normal,
        ],
    ):
        if not isinstance(prior_process.transition, statespace.Integrator):
            raise ValueError(
                "Please initialise a Gaussian filter with an Integrator (see `probnum.statespace`)"
            )

        self.prior_process = prior_process
        self.measurement_model = measurement_model

        self.sigma_squared_mle = 1.0
        self.with_smoothing = with_smoothing
        self.init_implementation = init_implementation
        super().__init__(ivp=ivp, order=prior_process.transition.ordint)

    @staticmethod
    def string_to_measurement_model(
        measurement_model_string, ivp, prior_process, measurement_noise_covariance=0.0
    ):
        """Construct a measurement model :math:`\\mathcal{N}(g(m), R)` for an ODE.

        Return a :class:`DiscreteGaussian` (either a :class:`DiscreteEKFComponent` or a `DiscreteUKFComponent`) that provides
        a tractable approximation of the transition densities based on the local defect of the ODE

        .. math:: g(m) = H_1 m(t) - f(t, H_0 m(t))

        and user-specified measurement noise covariance :math:`R`. Almost always, the measurement noise covariance is zero.

        EK0, EK1, and UK derive a tractable approximation of this intractable model with zeroth-order or first-order Taylor series approximations,
        respectively the unscented transform.
        """
        measurement_model_string = measurement_model_string.upper()

        # While "UK" is not available in probsolve_ivp (because it is not recommended)
        # It is an option in this function here, because there is no obvious reason to restrict
        # the options in this lower level function.
        choose_meas_model = {
            "EK0": filtsmooth.DiscreteEKFComponent.from_ode(
                ivp,
                prior=prior_process.transition,
                ek0_or_ek1=0,
                evlvar=measurement_noise_covariance,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            ),
            "EK1": filtsmooth.DiscreteEKFComponent.from_ode(
                ivp,
                prior=prior_process.transition,
                ek0_or_ek1=1,
                evlvar=measurement_noise_covariance,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            ),
            "UK": filtsmooth.DiscreteUKFComponent.from_ode(
                ivp,
                prior=prior_process.transition,
                evlvar=measurement_noise_covariance,
            ),
        }

        if measurement_model_string not in choose_meas_model.keys():
            raise ValueError("Type of measurement model not supported.")

        return choose_meas_model[measurement_model_string]

    # Construct an ODE solver from different initialisation methods.
    # The reason for implementing these via classmethods is that different
    # initialisation methods require different parameters.

    @classmethod
    def construct_with_rk_init(
        cls,
        ivp,
        prior_process,
        measurement_model,
        with_smoothing,
        init_h0=0.01,
        init_method="DOP853",
    ):
        """Create a Gaussian IVP filter that is initialised via
        :func:`initialize_odefilter_with_rk`."""

        def init_implementation(f, y0, t0, prior_process, df=None):
            return initialize_odefilter_with_rk(
                f=f,
                y0=y0,
                t0=t0,
                prior_process=prior_process,
                df=df,
                h0=init_h0,
                method=init_method,
            )

        return cls(
            ivp,
            prior_process,
            measurement_model,
            with_smoothing,
            init_implementation=init_implementation,
        )

    @classmethod
    def construct_with_taylormode_init(
        cls, ivp, prior_process, measurement_model, with_smoothing
    ):
        """Create a Gaussian IVP filter that is initialised via
        :func:`initialize_odefilter_with_taylormode`."""
        return cls(
            ivp,
            prior_process,
            measurement_model,
            with_smoothing,
            init_implementation=initialize_odefilter_with_taylormode,
        )

    def initialise(self):
        initrv = self.init_implementation(
            self.ivp.rhs,
            self.ivp.initrv.mean,
            self.ivp.t0,
            self.prior_process,
            self.ivp._jac,
        )

        return self.ivp.t0, initrv

    def step(self, t, t_new, current_rv):
        """Gaussian IVP filter step as nonlinear Kalman filtering with zero data."""

        # Read the diffusion matrix; required for calibration / error estimation
        discrete_dynamics = self.prior_process.transition.discretise(t_new - t)
        proc_noise_cov = discrete_dynamics.proc_noise_cov_mat
        proc_noise_cov_cholesky = discrete_dynamics.proc_noise_cov_cholesky

        # 1. Predict
        pred_rv, _ = self.prior_process.transition.forward_rv(
            rv=current_rv, t=t, dt=t_new - t
        )

        # 2. Measure
        meas_rv, info = self.measurement_model.forward_rv(
            rv=pred_rv, t=t_new, compute_gain=False
        )

        # 3. Estimate the diffusion (sigma squared)
        self.sigma_squared_mle = self._estimate_diffusion(meas_rv)

        # 3.1. Adjust the prediction covariance to include the diffusion
        pred_rv, _ = self.prior_process.transition.forward_rv(
            rv=current_rv, t=t, dt=t_new - t, _diffusion=self.sigma_squared_mle
        )

        # 3.2 Update the measurement covariance (measure again)
        meas_rv, info = self.measurement_model.forward_rv(
            rv=pred_rv, t=t_new, compute_gain=True
        )

        # 4. Update
        zero_data = np.zeros(meas_rv.mean.shape)
        filt_rv, _ = self.measurement_model.backward_realization(
            zero_data, pred_rv, rv_forwarded=meas_rv, gain=info["gain"]
        )

        # 5. Error estimate
        local_errors = self._estimate_local_error(
            pred_rv,
            t_new,
            self.sigma_squared_mle * proc_noise_cov,
            np.sqrt(self.sigma_squared_mle) * proc_noise_cov_cholesky,
        )
        err = np.linalg.norm(local_errors)

        return filt_rv, err

    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""

        kalman_posterior = filtsmooth.FilteringPosterior(
            times, rvs, self.prior_process.transition
        )

        return KalmanODESolution(kalman_posterior)

    def postprocess(self, odesol):
        """If specified (at initialisation), smooth the filter output."""
        if False:  # pylint: disable=using-constant-test
            # will become useful again for time-fixed diffusion models
            rvs = self._rescale(rvs)
        if self.with_smoothing is True:
            odesol = self._odesmooth(ode_solution=odesol)
        return odesol

    def _rescale(self, rvs):
        """Rescales covariances according to estimate sigma squared value."""
        rvs = [randvars.Normal(rv.mean, self.sigma_squared_mle * rv.cov) for rv in rvs]
        return rvs

    def _odesmooth(self, ode_solution, **kwargs):
        """Smooth out the ODE-Filter output.

        Be careful about the preconditioning: the GaussFiltSmooth object
        only knows the state space with changed coordinates!

        Parameters
        ----------
        filter_solution: ODESolution

        Returns
        -------
        smoothed_solution: ODESolution
        """
        smoothing_posterior = filtsmooth.Kalman(self.prior_process).smooth(
            ode_solution.kalman_posterior
        )
        return KalmanODESolution(smoothing_posterior)

    def _estimate_local_error(
        self,
        pred_rv,
        t_new,
        calibrated_proc_noise_cov,
        calibrated_proc_noise_cov_cholesky,
        **kwargs
    ):
        """Estimate the local errors.

        This corresponds to the approach in [1], implemented such that it is compatible
        with the EKF1 and UKF.

        References
        ----------
        .. [1] Schober, M., Särkkä, S. and Hennig, P..
            A probabilistic model for the numerical solution of initial
            value problems.
            Statistics and Computing, 2019.
        """
        local_pred_rv = randvars.Normal(
            pred_rv.mean,
            calibrated_proc_noise_cov,
            cov_cholesky=calibrated_proc_noise_cov_cholesky,
        )
        local_meas_rv, _ = self.measurement_model.forward_rv(rv=local_pred_rv, t=t_new)
        error = local_meas_rv.cov.diagonal()
        return np.sqrt(np.abs(error))

    def _estimate_diffusion(self, meas_rv):
        """Estimate the dynamic diffusion parameter sigma_squared.

        This corresponds to the approach in [1], implemented such that it is compatible
        with the EKF1 and UKF.

        References
        ----------
        .. [1] Schober, M., Särkkä, S. and Hennig, P..
            A probabilistic model for the numerical solution of initial
            value problems.
            Statistics and Computing, 2019.
        """
        std_like = meas_rv.cov_cholesky
        whitened_res = np.linalg.solve(std_like, meas_rv.mean)
        ssq = whitened_res @ whitened_res / meas_rv.size
        return ssq

    @property
    def prior(self):
        return self.prior_process.transition
