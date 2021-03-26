"""Gaussian IVP filtering and smoothing."""

from typing import Callable, Optional

import numpy as np
import scipy.linalg

from probnum import filtsmooth, randvars, statespace, utils

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
    dynamics_model :
        dynamics_model distribution.
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
    diffusion_model :
        Diffusion model. This determines which kind of calibration is used. We refer to Bosch et al. (2020) [1]_ for a survey.
    _repeat_after_calibration :
        Repeat the step after the initial calibration. Improves the result of each step, but slightly increases computational complexity.
        Optional. A default value is inferred from the type of diffusion_model model:
        if it is a `ConstantDiffusion`, this parameter is set to `False`;
        if it is a `PiecewiseConstantDiffusion`, this parameter is set to `True`.
        Other combinations are possible, but we recommend to stick to the defaults here.
        If you decide to change this argument, it happens at your own risk!
    _reference_coordinates :
        Use this state as a reference state to compute the normalized error estimate.
        Optional. Default is 0 (which amounts to the usual reference state for ODE solvers).
        Another reasonable choice could be 1, but use this at your own risk!

    References
    ----------
    .. [1] Bosch, N., and Hennig, P., and Tronarp, F..
        Calibrated Adaptive Probabilistic ODE Solvers.
        2021.
    """

    def __init__(
        self,
        ivp: IVP,
        dynamics_model: statespace.Integrator,
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
        initrv: Optional[randvars.Normal] = None,
        diffusion_model: Optional[Diffusion] = None,
        _repeat_after_calibration: Optional[bool] = None,
        _reference_coordinates: Optional[int] = 0,
    ):

        # Raise a comprehensible error if the wrong models are passed.
        if not isinstance(measurement_model, filtsmooth.DiscreteEKFComponent):
            raise TypeError(
                "Please initialize a Gaussian ODE filter with an EKF component as a measurement model."
            )
        if not isinstance(dynamics_model, statespace.Integrator):
            raise TypeError(
                "Please initialize a Gaussian ODE filter with an Integrator as a dynamics_model"
            )

        if initrv is None:
            d = dynamics_model.dimension
            scale = 1e6
            initrv = randvars.Normal(
                mean=np.zeros(d),
                cov=scale * np.eye(d),
                cov_cholesky=np.sqrt(scale) * np.eye(d),
            )
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv

        self.with_smoothing = with_smoothing
        self.init_implementation = init_implementation
        super().__init__(ivp=ivp, order=dynamics_model.ordint)

        self.sigma_squared_mle = 1.0
        # Set up the diffusion_model style: constant or piecewise constant.
        self.diffusion_model = (
            PiecewiseConstantDiffusion() if diffusion_model is None else diffusion_model
        )

        if _repeat_after_calibration is None:
            if isinstance(self.diffusion_model, PiecewiseConstantDiffusion):
                self._repeat_after_calibration = True
            else:
                self._repeat_after_calibration = False
        else:
            self._repeat_after_calibration = _repeat_after_calibration

        self._reference_coordinates = _reference_coordinates

    # Construct an ODE solver from different initialisation methods.
    # The reason for implementing these via classmethods is that different
    # initialisation methods require different parameters.

    @classmethod
    def construct_with_rk_init(
        cls,
        ivp,
        dynamics_model,
        measurement_model,
        with_smoothing,
        initrv=None,
        diffusion_model=None,
        _repeat_after_calibration=True,
        _reference_coordinates=0,
        init_h0=0.01,
        init_method="DOP853",
    ):
        """Create a Gaussian IVP filter that is initialised via
        :func:`initialize_odefilter_with_rk`."""

        def init_implementation(f, y0, t0, dynamics_model, initrv, df=None):
            return initialize_odefilter_with_rk(
                f=f,
                y0=y0,
                t0=t0,
                prior=dynamics_model,
                initrv=initrv,
                df=df,
                h0=init_h0,
                method=init_method,
            )

        return cls(
            ivp,
            dynamics_model,
            measurement_model,
            with_smoothing,
            init_implementation=init_implementation,
            initrv=initrv,
            diffusion_model=diffusion_model,
            _repeat_after_calibration=_repeat_after_calibration,
            _reference_coordinates=_reference_coordinates,
        )

    @classmethod
    def construct_with_taylormode_init(
        cls,
        ivp,
        dynamics_model,
        measurement_model,
        with_smoothing,
        initrv=None,
        diffusion_model=None,
        _repeat_after_calibration=True,
        _reference_coordinates=0,
    ):
        """Create a Gaussian IVP filter that is initialised via
        :func:`initialize_odefilter_with_taylormode`."""
        return cls(
            ivp,
            dynamics_model,
            measurement_model,
            with_smoothing,
            init_implementation=initialize_odefilter_with_taylormode,
            initrv=initrv,
            diffusion_model=diffusion_model,
            _repeat_after_calibration=_repeat_after_calibration,
            _reference_coordinates=_reference_coordinates,
        )

    def initialise(self):
        initrv = self.init_implementation(
            self.ivp.rhs,
            self.ivp.initrv.mean,
            self.ivp.t0,
            self.dynamics_model,
            self.initrv,
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
        Both :math:`z(t + \Delta t)` and :math:`\hat z(t + \Delta t)` give rise to a reasonable diffusion_model estimate.
        Which one to use is handled by the ``Diffusion`` attribute of the solver.
        At this point already we can compute a local error estimate of the current step.

        4. Once the diffusion_model is chosen, there are two options:

            4.1. If the calibrated diffusion_model is used to repeat prediction and measurement step, we compute new values for
            :math:`x(t+\Delta t) \,|\, x(t)` and for :math:`z(t)`. While this adds computational expense, the uncertainties of these
            values will be calibrated much better than in the other scenario:

            4.2. If the calibrated diffusion_model is only used for a post-hoc rescaling of the covariances, we only need to assemble
            the prediction  :math:`x(t+\Delta t) | x(t)`

            .. math::
                x(t+\Delta t) \,|\, x(t) \sim \mathcal{N}(\Phi(\Delta t) m(t), \Phi(\Delta t) P(t) \Phi(\Delta t)^\top + Q(\Delta t))

            from quantities that have been computed above.

        5. With the results of either 4.1. or 4.2. (which both return a predicted RV and a measured RV),
        we finally compute the Kalman update and return the result. Recall that the error estimate has been computed in the third step.
        """

        # Read off system matrices; required for calibration / error estimation
        # Use only where a full call to forward_*() would be too costly.
        # We use the mathematical symbol `Phi` (and later, `H`), because this makes it easier to read for us.
        # The system matrix H of the measurement model can be accessed after the first forward_*,
        # therefore we read it off further below.
        dt = t_new - t
        discrete_dynamics = self.dynamics_model.discretise(dt)
        Phi = discrete_dynamics.state_trans_mat

        # Split the current RV into a deterministic part and a noisy part.
        # This split is necessary for efficient calibration; see docstring.
        error_free_state = current_rv.mean.copy()
        noisy_component = randvars.Normal(
            mean=np.zeros(current_rv.shape),
            cov=current_rv.cov.copy(),
            cov_cholesky=current_rv.cov_cholesky.copy(),
        )

        # Compute the measurements for the error-free component
        pred_rv_error_free, _ = self.dynamics_model.forward_realization(
            error_free_state, t=t, dt=dt
        )
        meas_rv_error_free, _ = self.measurement_model.forward_rv(
            pred_rv_error_free, t=t
        )
        H = self.measurement_model.linearized_model.state_trans_mat_fun(t=t)

        # Compute the measurements for the full components.
        # Since the means of noise-free and noisy measurements coincide,
        # we manually update only the covariance.
        # The first two are only matrix square-roots and will be turned into proper Cholesky factors below.
        pred_sqrtm = Phi @ noisy_component.cov_cholesky
        meas_sqrtm = H @ pred_sqrtm
        full_meas_cov_cholesky = utils.linalg.cholesky_update(
            meas_rv_error_free.cov_cholesky, meas_sqrtm
        )
        full_meas_cov = full_meas_cov_cholesky @ full_meas_cov_cholesky.T
        meas_rv = randvars.Normal(
            mean=meas_rv_error_free.mean,
            cov=full_meas_cov,
            cov_cholesky=full_meas_cov_cholesky,
        )

        # Estimate local diffusion_model and error
        diffusion_error_free = self.diffusion_model.calibrate_locally(
            meas_rv_error_free
        )
        diffusion_full = self.diffusion_model.calibrate_locally(meas_rv)
        diffusion_for_calibration = self.diffusion_model.update_current_information(
            diffusion_full, diffusion_error_free, t_new
        )
        local_errors = np.sqrt(diffusion_for_calibration) * np.sqrt(
            np.diag(meas_rv_error_free.cov)
        )

        # Either re-predict and re-measure with improved calibration or let the predicted RV catch up.
        if self._repeat_after_calibration:
            pred_rv, _ = self.dynamics_model.forward_rv(
                rv=current_rv, t=t, dt=dt, _diffusion=diffusion_for_calibration
            )
            meas_rv, info = self.measurement_model.forward_rv(
                rv=pred_rv, t=t_new, compute_gain=True
            )
            gain = info["gain"]
        else:
            # Combine error-free and noisy-component predictions into a full prediction.
            # (The measurement has been updated already.)
            full_pred_cov_cholesky = utils.linalg.cholesky_update(
                pred_rv_error_free.cov_cholesky, pred_sqrtm
            )
            full_pred_cov = full_pred_cov_cholesky @ full_pred_cov_cholesky.T
            pred_rv = randvars.Normal(
                mean=pred_rv_error_free.mean,
                cov=full_pred_cov,
                cov_cholesky=full_pred_cov_cholesky,
            )

            # Gain needs manual catching up, too.
            crosscov = full_pred_cov @ H.T
            gain = scipy.linalg.cho_solve((meas_rv.cov_cholesky, True), crosscov.T).T

        # 4. Update
        zero_data = np.zeros(meas_rv.mean.shape)
        filt_rv, _ = self.measurement_model.backward_realization(
            zero_data, pred_rv, rv_forwarded=meas_rv, gain=gain
        )

        proj = self.dynamics_model.proj2coord(coord=self._reference_coordinates)
        reference_state = np.maximum(proj @ filt_rv.mean, proj @ current_rv.mean)
        return filt_rv, local_errors, reference_state

    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""

        kalman_posterior = filtsmooth.FilteringPosterior(
            times, rvs, self.dynamics_model
        )

        return KalmanODESolution(kalman_posterior)

    def postprocess(self, odesol):
        """If specified (at initialisation), smooth the filter output."""
        locations = odesol.kalman_posterior.locations
        rv_list = self.diffusion_model.calibrate_all_states(
            odesol.kalman_posterior.states, locations
        )

        kalman_posterior = filtsmooth.FilteringPosterior(
            locations, rv_list, self.dynamics_model
        )

        if self.with_smoothing is True:
            rv_list = self.dynamics_model.smooth_list(rv_list, locations)
            kalman_posterior = filtsmooth.SmoothingPosterior(
                locations,
                rv_list,
                self.dynamics_model,
                filtering_posterior=kalman_posterior,
            )

        return KalmanODESolution(kalman_posterior)

    @staticmethod
    def string_to_measurement_model(
        measurement_model_string, ivp, dynamics_model, measurement_noise_covariance=0.0
    ):
        """Construct a measurement model :math:`\\mathcal{N}(g(m), R)` for an ODE.

        Return a :class:`DiscreteGaussian` (:class:`DiscreteEKFComponent`) that provides
        a tractable approximation of the transition densities based on the local defect of the ODE

        .. math:: g(m) = H_1 m(t) - f(t, H_0 m(t))

        and user-specified measurement noise covariance :math:`R`. Almost always, the measurement noise covariance is zero.

        Compute either type filter, each with a different interpretation of the Jacobian :math:`J_g`:

        - EKF0 thinks :math:`J_g(m) = H_1`
        - EKF1 thinks :math:`J_g(m) = H_1 - J_f(t, H_0 m(t)) H_0^\\top`
        """
        measurement_model_string = measurement_model_string.upper()

        # While "UK" is not available in probsolve_ivp (because it is not recommended)
        # It is an option in this function here, because there is no obvious reason to restrict
        # the options in this lower level function.
        choose_meas_model = {
            "EK0": filtsmooth.DiscreteEKFComponent.from_ode(
                ivp,
                prior=dynamics_model,
                ek0_or_ek1=0,
                evlvar=measurement_noise_covariance,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            ),
            "EK1": filtsmooth.DiscreteEKFComponent.from_ode(
                ivp,
                prior=dynamics_model,
                ek0_or_ek1=1,
                evlvar=measurement_noise_covariance,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            ),
        }

        if measurement_model_string not in choose_meas_model.keys():
            raise ValueError("Type of measurement model not supported.")

        return choose_meas_model[measurement_model_string]
