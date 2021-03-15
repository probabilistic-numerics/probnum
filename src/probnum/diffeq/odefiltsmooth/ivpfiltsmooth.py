import typing

import numpy as np
import scipy.linalg

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum import utils
from probnum.random_variables import Normal

from ..ode import IVP
from ..odesolver import ODESolver
from .diffusions import Diffusion, PiecewiseConstantDiffusion
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
    ivp :
        Initial value problem to be solved.
    prior :
        Prior distribution.
    measurement_model :
        Linearized ODE measurement model. Must be a `DiscreteEKFComponent`.
    with_smoothing :
        To smooth after the solve or not to smooth after the solve.
    init_implementation :
        Initialization algorithm. Either via Scipy (``initialize_odefilter_with_rk``) or via Taylor-mode AD (``initialize_odefilter_with_taylormode``).
        For more convenient construction, consider :func:`GaussianIVPFilter.construct_with_rk_init` and :func:`GaussianIVPFilter.construct_with_taylormode_init`.
    initrv :
        Initial random variable to be used by the filter. Optional. Default is `None`, which amounts to choosing a standard-normal initial RV.
        As part of `GaussianIVPFilter.initialise()` (called in `GaussianIVPFilter.solve()`), this variable is improved upon with the help of the
        initialisation algorithm. The influence of this choice on the posterior may vary depending on the initialization strategy, but is almost always weak.
    diffusion :
        Diffusion model.
    re_predict_with_calibrated_diffusion :
        Re-start the step after the initial calibration. Improves the result of each step, but slightly increases computational complexity.
    """

    def __init__(
        self,
        ivp: IVP,
        prior: pnss.Integrator,
        measurement_model: pnfs.DiscreteEKFComponent,
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
        diffusion: typing.Optional[Diffusion] = None,
        re_predict_with_calibrated_diffusion: typing.Optional[bool] = True,
    ):
        if initrv is None:
            initrv = Normal(
                np.zeros(prior.dimension),
                np.eye(prior.dimension),
                cov_cholesky=np.eye(prior.dimension),
            )

        if not isinstance(measurement_model, pnfs.DiscreteEKFComponent):
            raise TypeError

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
        r"""Gaussian IVP filter step as nonlinear Kalman filtering with zero data.

        It goes as follows:

        1. The current state :math:`x(t) \sim \mathcal{N}(m(t), P(t))` is split into a deterministic component
        and a noisy component,

        .. math::
            x(t) = m(t) + p(t), \quad p(t) \sim \mathcal{N}(0, P(t))

        which is required for accurate calibration and error estimation: in order to only work with the local error,
        ODE solvers often assume that the error at the previous step is zero.

        2. The deterministic component is propagated through dynamics model and measurement model

        .. math::
            \hat{z}(t + \Delta t) \sim \mathcal{N}(H \Phi(\Delta t) m(t), H Q(\Delta t) H^\top)

        which is a random variable that estimates the expected local defect :math:`\dot y - f(y)`.

        3. The counterpart of :math:`\hat{z}` in the (much more likely) interpretation of an erronous previous state
        (recall that :math:`\hat z` was based on the interpretation of an error-free previous state) is computed as,

        .. math::
            z(t + \Delta t) \sim \mathcal{N}(\mathbb{E}[\hat{z}(t + \Delta t)],
            \mathbb{C}[\hat{z}(t + \Delta t)] + H \Phi(\Delta t) P(t) \Phi(\Delta t)^\top H^\top ),

        which acknowledges the covariance :math:`P(t)` of the previous state.
        :math:`\mathbb{E}` is the mean, and :math:`\mathbb{C}` is the covariance.
        Both :math:`z(t + \Delta t)` and :math:`\hat z(t + \Delta t)` give rise to a reasonable diffusion estimate.
        Which one to use is handled by the ``Diffusion`` attribute of the solver.
        At this point already we can compute a local error estimate of the current step.

        4. Once the diffusion is chosen, there are two options:

            4.1. If the calibrated diffusion is used to repeat prediction and measurement step, we compute new values for
            :math:`x(t+\Delta t) \,|\, x(t)` and for :math:`z(t)`. While this adds computational expense, the uncertainties of these
            values will be calibrated much better than in the other scenario:

            4.2. If the calibrated diffusion is only used for a post-hoc rescaling of the covariances, we only need to assemble
            the prediction  :math:`x(t+\Delta t) | x(t)`

            .. math::
                x(t+\Delta t) \,|\, x(t) \sim \mathcal{N}(\Phi(\Delta t) m(t), \Phi(\Delta t) P(t) \Phi(\Delta t)^\top + Q(\Delta t))

            from quantities that have been computed above.

        5. With the results of either 4.1. or 4.2. (which both return a predicted RV and a measured RV),
        we finally compute the Kalman update and return the result. Recall that the error estimate has been computed in the third step.
        """

        # Read the diffusion matrix; required for calibration / error estimation
        dt = t_new - t
        discrete_dynamics = self.gfilt.dynamics_model.discretise(dt)
        noise_cov = discrete_dynamics.proc_noise_cov_mat
        noise_cov_cholesky = discrete_dynamics.proc_noise_cov_cholesky

        state_transition = discrete_dynamics.state_trans_mat

        # Split the current RV into a deterministic part and a noisy part.
        # This split is necessary for efficient calibration.
        error_free_state = current_rv.mean.copy()
        noisy_component = Normal(
            mean=np.zeros(current_rv.shape),
            cov=current_rv.cov.copy(),
            cov_cholesky=current_rv.cov_cholesky.copy(),
        )

        # Compute the measurements for the error-free component
        pred_rv_error_free, _ = self.gfilt.dynamics_model.forward_realization(
            error_free_state, t=t, dt=dt
        )
        meas_rv_error_free, _ = self.gfilt.measurement_model.forward_rv(
            pred_rv_error_free, t=t
        )

        # Compute the measurements for the noisy components.
        # This only depends on the deterministic part of the transition.
        # The resulting measurement RV has zero mean.
        pred_rv_noisy_component = _apply_state_trans(state_transition, noisy_component)
        meas_rv_noisy_component, _ = self.gfilt.measurement_model.forward_rv(
            pred_rv_noisy_component, t=t
        )

        # Update the measurement RV.
        # The prediction RV may be updated again below (which can be combined with the current formula),
        # so we delay this computation for a bit.
        full_meas_cov_cholesky = utils.linalg.cholesky_update(
            meas_rv_error_free.cov_cholesky, meas_rv_noisy_component.cov_cholesky
        )
        full_meas_cov = full_meas_cov_cholesky @ full_meas_cov_cholesky.T
        meas_rv = Normal(
            mean=meas_rv_error_free.mean,
            cov=full_meas_cov,
            cov_cholesky=full_meas_cov_cholesky,
        )

        # Estimate local diffusions
        local_squared_diffusion_error_free = self.diffusion.calibrate_locally(
            meas_rv_error_free
        )
        local_squared_diffusion_full = self.diffusion.calibrate_locally(meas_rv)
        diffusion_for_calibration = self.diffusion.update_current_information(
            local_squared_diffusion_full, local_squared_diffusion_error_free, t_new
        )

        if self.re_predict_with_calibrated_diffusion:
            # Re-predict and re-measure with improved calibration
            pred_rv, _ = self.gfilt.dynamics_model.forward_rv(
                rv=current_rv, t=t, dt=dt, _diffusion=diffusion_for_calibration
            )
            meas_rv, info = self.gfilt.measurement_model.forward_rv(
                rv=pred_rv, t=t_new, compute_gain=True
            )
            gain = info["gain"]
        else:
            # Combine error-free and noisy-component predictions into a full prediction.
            # (The measurement has been updated already.)
            full_pred_cov_cholesky = utils.linalg.cholesky_update(
                pred_rv_error_free.cov_cholesky, pred_rv_noisy_component.cov_cholesky
            )
            full_pred_cov = full_pred_cov_cholesky @ full_pred_cov_cholesky.T
            pred_rv = Normal(
                mean=pred_rv_error_free.mean,
                cov=full_pred_cov,
                cov_cholesky=full_pred_cov_cholesky,
            )

            # Gain needs manual catching up
            H = self.gfilt.measurement_model.linearized_model.state_trans_mat_fun(t=t)
            crosscov = full_pred_cov @ H.T
            gain = scipy.linalg.cho_solve((meas_rv.cov_cholesky, True), crosscov.T).T

        # 4. Update
        zero_data = np.zeros(meas_rv.mean.shape)
        filt_rv, _ = self.gfilt.measurement_model.backward_realization(
            zero_data, pred_rv, rv_forwarded=meas_rv, gain=gain
        )

        # 5. Error estimate
        local_errors = self._estimate_local_error(
            pred_rv,
            t_new,
            diffusion_for_calibration * noise_cov,
            np.sqrt(diffusion_for_calibration) * noise_cov_cholesky,
        )
        err = np.linalg.norm(local_errors)

        return filt_rv, err

    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""

        rvs = self.diffusion.calibrate_all_states(rvs, times)

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


def _apply_state_trans(H, rv):

    # There is no way of checking whether `rv` has its Cholesky factor computed already or not.
    # Therefore, since we need to update the Cholesky factor for square-root filtering,
    # we also update the Cholesky factor for non-square-root algorithms here,
    # which implies additional cost.
    # See Issues #319 and #329.
    # When they are resolved, this function here will hopefully be superfluous.

    new_mean = H @ rv.mean
    new_cov_cholesky = utils.linalg.cholesky_update(H @ rv.cov_cholesky)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T

    return Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky)
