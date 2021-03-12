import typing

import numpy as np

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum.random_variables import Normal

from ..ode import IVP
from ..odesolver import ODESolver
from .diffusions import PiecewiseConstantDiffusion
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
    prior
        Prior distribution.
    measurement_model
        ODE measurement model.
    with_smoothing
        To smooth after the solve or not to smooth after the solve.
    init_implementation
        Initialization algorithm. Either via Scipy (``initialize_odefilter_with_rk``) or via Taylor-mode AD (``initialize_odefilter_with_taylormode``).
        For more convenient construction, consider :func:`GaussianIVPFilter.construct_with_rk_init` and :func:`GaussianIVPFilter.construct_with_taylormode_init`.
    initrv
        Initial random variable to be used by the filter. Optional. Default is `None`, which amounts to choosing a standard-normal initial RV.
        As part of `GaussianIVPFilter.initialise()` (called in `GaussianIVPFilter.solve()`), this variable is improved upon with the help of the
        initialisation algorithm. The influence of this choice on the posterior may vary depending on the initialization strategy, but is almost always weak.
    """

    def __init__(
        self,
        ivp: IVP,
        prior: pnss.Integrator,
        measurement_model: pnss.DiscreteGaussian,
        with_smoothing: bool,
        init_implementation: typing.Callable[
            [
                typing.Callable,
                np.ndarray,
                float,
                pnss.Integrator,
                Normal,
                typing.Optional[typing.Callable],
            ],
            Normal,
        ],
        initrv: typing.Optional[Normal] = None,
        diffusion=None,
        re_predict_with_calibrated_diffusion=True,
    ):
        if initrv is None:
            initrv = Normal(
                np.zeros(prior.dimension),
                np.eye(prior.dimension),
                cov_cholesky=np.eye(prior.dimension),
            )

        self.gfilt = pnfs.Kalman(
            dynamics_model=prior, measurement_model=measurement_model, initrv=initrv
        )

        if not isinstance(prior, pnss.Integrator):
            raise ValueError(
                "Please initialise a Gaussian filter with an Integrator (see `probnum.statespace`)"
            )
        self.with_smoothing = with_smoothing
        self.init_implementation = init_implementation
        super().__init__(ivp=ivp, order=prior.ordint)

        # Set up the diffusion style: constant or piecewise constant.
        self.diffusion = (
            PiecewiseConstantDiffusion() if diffusion is None else diffusion
        )
        self.re_predict_with_calibrated_diffusion = re_predict_with_calibrated_diffusion

    # Construct an ODE solver from different initialisation methods.
    # The reason for implementing these via classmethods is that different
    # initialisation methods require different parameters.

    @classmethod
    def construct_with_rk_init(
        cls,
        ivp,
        prior,
        measurement_model,
        with_smoothing,
        initrv=None,
        diffusion=None,
        re_predict_with_calibrated_diffusion=True,
        init_h0=0.01,
        init_method="DOP853",
    ):
        """Create a Gaussian IVP filter that is initialised via
        :func:`initialize_odefilter_with_rk`."""

        def init_implementation(f, y0, t0, prior, initrv, df=None):
            return initialize_odefilter_with_rk(
                f=f,
                y0=y0,
                t0=t0,
                prior=prior,
                initrv=initrv,
                df=df,
                h0=init_h0,
                method=init_method,
            )

        return cls(
            ivp,
            prior,
            measurement_model,
            with_smoothing,
            init_implementation=init_implementation,
            initrv=initrv,
            diffusion=diffusion,
            re_predict_with_calibrated_diffusion=re_predict_with_calibrated_diffusion,
        )

    @classmethod
    def construct_with_taylormode_init(
        cls,
        ivp,
        prior,
        measurement_model,
        with_smoothing,
        initrv=None,
        diffusion=None,
        re_predict_with_calibrated_diffusion=True,
    ):
        """Create a Gaussian IVP filter that is initialised via
        :func:`initialize_odefilter_with_taylormode`."""
        return cls(
            ivp,
            prior,
            measurement_model,
            with_smoothing,
            init_implementation=initialize_odefilter_with_taylormode,
            initrv=initrv,
            diffusion=diffusion,
            re_predict_with_calibrated_diffusion=re_predict_with_calibrated_diffusion,
        )

    def initialise(self):
        initrv = self.init_implementation(
            self.ivp.rhs,
            self.ivp.initrv.mean,
            self.ivp.t0,
            self.gfilt.dynamics_model,
            self.gfilt.initrv,
            self.ivp._jac,
        )

        return self.ivp.t0, initrv

    def step(self, t, t_new, current_rv):
        """Gaussian IVP filter step as nonlinear Kalman filtering with zero data."""

        # Read the diffusion matrix; required for calibration / error estimation
        discrete_dynamics = self.gfilt.dynamics_model.discretise(t_new - t)
        proc_noise_cov = discrete_dynamics.proc_noise_cov_mat
        proc_noise_cov_cholesky = discrete_dynamics.proc_noise_cov_cholesky

        # 1. Predict
        pred_rv, _ = self.gfilt.dynamics_model.forward_rv(
            rv=current_rv, t=t, dt=t_new - t
        )

        # 2. Measure
        compute_gain = not self.re_predict_with_calibrated_diffusion
        meas_rv, info = self.gfilt.measurement_model.forward_rv(
            rv=pred_rv, t=t_new, compute_gain=compute_gain
        )

        # 3. Estimate the diffusion (sigma squared)
        local_squared_diffusion = self.diffusion.calibrate_locally(meas_rv)
        self.diffusion.update_current_information(local_squared_diffusion, t_new)

        if self.re_predict_with_calibrated_diffusion:
            # 3.1. Adjust the prediction covariance to include the diffusion
            pred_rv, _ = self.gfilt.dynamics_model.forward_rv(
                rv=current_rv, t=t, dt=t_new - t, _diffusion=local_squared_diffusion
            )

            # 3.2 Update the measurement covariance (measure again)
            meas_rv, info = self.gfilt.measurement_model.forward_rv(
                rv=pred_rv, t=t_new, compute_gain=True
            )

        # 4. Update
        zero_data = np.zeros(meas_rv.mean.shape)
        filt_rv, _ = self.gfilt.measurement_model.backward_realization(
            zero_data, pred_rv, rv_forwarded=meas_rv, gain=info["gain"]
        )

        # 5. Error estimate
        local_errors = self._estimate_local_error(
            pred_rv,
            t_new,
            local_squared_diffusion * proc_noise_cov,
            np.sqrt(local_squared_diffusion) * proc_noise_cov_cholesky,
        )
        err = np.linalg.norm(local_errors)

        return filt_rv, err

    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""

        rvs = self.diffusion.postprocess_states(rvs, times)

        kalman_posterior = pnfs.FilteringPosterior(
            times, rvs, self.gfilt.dynamics_model
        )

        return KalmanODESolution(kalman_posterior)

    def postprocess(self, odesol):
        """If specified (at initialisation), smooth the filter output."""

        if self.with_smoothing is True:
            smoothing_posterior = self.gfilt.smooth(odesol.kalman_posterior)
            odesol = KalmanODESolution(smoothing_posterior)

        return odesol

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
        local_pred_rv = Normal(
            pred_rv.mean,
            calibrated_proc_noise_cov,
            cov_cholesky=calibrated_proc_noise_cov_cholesky,
        )
        local_meas_rv, _ = self.gfilt.measure(local_pred_rv, t_new)
        error = local_meas_rv.cov.diagonal()
        return np.sqrt(np.abs(error))

    @staticmethod
    def string_to_measurement_model(
        measurement_model_string, ivp, prior, measurement_noise_covariance=0.0
    ):
        """Construct a measurement model :math:`\\mathcal{N}(g(m), R)` for an ODE.

        Return a :class:`DiscreteGaussian` (either a :class:`DiscreteEKFComponent` or a `DiscreteUKFComponent`) that provides
        a tractable approximation of the transition densities based on the local defect of the ODE

        .. math:: g(m) = H_1 m(t) - f(t, H_0 m(t))

        and user-specified measurement noise covariance :math:`R`. Almost always, the measurement noise covariance is zero.

        Compute either type filter, each with a different interpretation of the Jacobian :math:`J_g`:

        - EKF0 thinks :math:`J_g(m) = H_1`
        - EKF1 thinks :math:`J_g(m) = H_1 - J_f(t, H_0 m(t)) H_0^\\top`
        - UKF thinks: ''What is a Jacobian?'' and uses the unscented transform to compute a tractable approximation of the transition densities.
        """
        measurement_model_string = measurement_model_string.upper()

        # While "UK" is not available in probsolve_ivp (because it is not recommended)
        # It is an option in this function here, because there is no obvious reason to restrict
        # the options in this lower level function.
        choose_meas_model = {
            "EK0": pnfs.DiscreteEKFComponent.from_ode(
                ivp,
                prior=prior,
                ek0_or_ek1=0,
                evlvar=measurement_noise_covariance,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            ),
            "EK1": pnfs.DiscreteEKFComponent.from_ode(
                ivp,
                prior=prior,
                ek0_or_ek1=1,
                evlvar=measurement_noise_covariance,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            ),
            "UK": pnfs.DiscreteUKFComponent.from_ode(
                ivp,
                prior,
                evlvar=measurement_noise_covariance,
            ),
        }

        if measurement_model_string not in choose_meas_model.keys():
            raise ValueError("Type of measurement model not supported.")

        return choose_meas_model[measurement_model_string]
