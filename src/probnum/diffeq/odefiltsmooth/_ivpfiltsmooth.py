"""Gaussian IVP filtering and smoothing."""

from typing import Optional

import numpy as np
import scipy.linalg

from probnum import filtsmooth, randprocs, randvars, utils
from probnum.diffeq import _odesolver, _odesolver_state, stepsize
from probnum.diffeq.odefiltsmooth import (
    _kalman_odesolution,
    approx_strategies,
    information_operators,
    initialization_routines,
)


class GaussianIVPFilter(_odesolver.ODESolver):
    """Probabilistic ODE solver based on Gaussian filtering and smoothing.

    This is based on continuous-discrete Gaussian filtering.

    Note: this is specific for IVPs and does not apply without
    further considerations to, e.g., BVPs.

    Parameters
    ----------
    ivp :
        Initial value problem to be solved.
    prior_process
        Prior Gauss-Markov process.
    measurement_model
        ODE measurement model.
    with_smoothing
        To smooth after the solve or not to smooth after the solve.
    initialization_routine :
        Initialization algorithm.
        Either via fitting the prior to a few steps of a Runge-Kutta method (:class:`RungeKuttaInitialization`)
        or via Taylor-mode automatic differentiation (:class:``TaylorModeInitialization``).
    diffusion_model :
        Diffusion model. This determines which kind of calibration is used. We refer to Bosch et al. (2020) [1]_ for a survey.
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
        steprule: stepsize.StepRule,
        prior_process: randprocs.markov.MarkovProcess,
        information_operator: information_operators.ODEInformationOperator,
        approx_strategy: approx_strategies.ApproximationStrategy,
        with_smoothing: bool,
        initialization_routine: initialization_routines.InitializationRoutine,
        diffusion_model: Optional[randprocs.markov.continuous.Diffusion] = None,
        _reference_coordinates: Optional[int] = 0,
    ):
        if not isinstance(
            prior_process.transition,
            randprocs.markov.integrator.IntegratorTransition,
        ):
            raise ValueError(
                "Please initialise a Gaussian filter with an Integrator (see `probnum.randprocs.markov.integrator`)"
            )

        self.prior_process = prior_process

        self.information_operator = information_operator
        self.approx_strategy = approx_strategy

        # Filled in in initialize(), once the ODE has been seen.
        self.measurement_model = None

        self.sigma_squared_mle = 1.0
        self.with_smoothing = with_smoothing
        self.initialization_routine = initialization_routine
        super().__init__(
            steprule=steprule, order=prior_process.transition.num_derivatives
        )

        # Set up the diffusion_model style: constant or piecewise constant.
        self.diffusion_model = (
            randprocs.markov.continuous.PiecewiseConstantDiffusion(
                t0=prior_process.initarg
            )
            if diffusion_model is None
            else diffusion_model
        )

        # Once the diffusion has been calibrated, the covariance can either
        # be calibrated after each step, or all states can be calibrated
        # with a global diffusion estimate. The choices here depend on the
        # employed diffusion model.
        is_dynamic = isinstance(
            self.diffusion_model, randprocs.markov.continuous.PiecewiseConstantDiffusion
        )
        self._calibrate_at_each_step = is_dynamic
        self._calibrate_all_states_post_hoc = not self._calibrate_at_each_step

        # Normalize the error estimate with the values from the 0th state
        # or from any other state.
        self._reference_coordinates = _reference_coordinates

    def initialize(self, ivp):
        self.information_operator.incorporate_ode(ode=ivp)
        self.measurement_model = self.approx_strategy(
            self.information_operator
        ).as_transition()

        initrv = self.initialization_routine(
            ivp=ivp,
            prior_process=self.prior_process,
        )
        state = _odesolver_state.ODESolverState(
            ivp=ivp, t=ivp.t0, rv=initrv, error_estimate=None, reference_state=None
        )
        return state

    def attempt_step(self, state, dt):
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

        4. Depending on the diffusion model, there are two options now:

            4.1. For a piecewise constant diffusion, the covariances are calibrated locally -- that is, at each step.
            In this case we update the predicted covariance and measured covariance with the most recent diffusion estimate.

            4.2. For a constant diffusion, the calibration happens post hoc, and the only step that is carried out here is an assembly
            of the full predicted random variable (up to now, only its parts were available).

        5. With the results of either 4.1. or 4.2. (which both return a predicted RV and a measured RV),
        we finally compute the Kalman update and return the result. Recall that the error estimate has been computed in the third step.
        """

        # Read off system matrices; required for calibration / error estimation
        # Use only where a full call to forward_*() would be too costly.
        # We use the mathematical symbol `Phi` (and later, `H`), because this makes it easier to read for us.
        # The system matrix H of the measurement model can be accessed after the first forward_*,
        # therefore we read it off further below.
        discrete_dynamics = self.prior_process.transition.discretise(dt)
        Phi = discrete_dynamics.state_trans_mat

        # Split the current RV into a deterministic part and a noisy part.
        # This split is necessary for efficient calibration; see docstring.
        error_free_state = state.rv.mean.copy()
        noisy_component = randvars.Normal(
            mean=np.zeros(state.rv.shape),
            cov=state.rv.cov.copy(),
            cov_cholesky=state.rv.cov_cholesky.copy(),
        )

        # Compute the measurements for the error-free component
        pred_rv_error_free, _ = self.prior_process.transition.forward_realization(
            error_free_state, t=state.t, dt=dt
        )
        meas_rv_error_free, _ = self.measurement_model.forward_rv(
            pred_rv_error_free, t=state.t + dt
        )
        H = self.measurement_model.linearized_model.state_trans_mat_fun(t=state.t + dt)

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
        local_diffusion = self.diffusion_model.estimate_locally(
            meas_rv=meas_rv,
            meas_rv_assuming_zero_previous_cov=meas_rv_error_free,
            t=state.t + dt,
        )

        local_errors = np.sqrt(local_diffusion) * np.sqrt(
            np.diag(meas_rv_error_free.cov)
        )
        proj = self.prior_process.transition.proj2coord(
            coord=self._reference_coordinates
        )
        reference_values = np.abs(proj @ state.rv.mean)

        # Overwrite the acceptance/rejection step here, because we need control over
        # Appending or not appending the diffusion (and because the computations below
        # are sufficiently costly such that skipping them here will have a positive impact).
        internal_norm = self.steprule.errorest_to_norm(
            errorest=local_errors,
            reference_state=reference_values,
        )

        if not self.steprule.is_accepted(internal_norm):
            t_new = state.t + dt
            new_rv = randvars.Normal(
                mean=state.rv.mean.copy(),
                cov=state.rv.cov.copy(),
                cov_cholesky=state.rv.cov_cholesky.copy(),
            )
            state = _odesolver_state.ODESolverState(
                ivp=state.ivp,
                rv=np.nan * new_rv,
                t=t_new,
                error_estimate=local_errors,
                reference_state=reference_values,
            )

            return state

        # Now we can be certain that the step will be accepted. In this case
        # we update the diffusion model and continue the rest of the iteration.,
        self.diffusion_model.update_in_place(local_diffusion, t=state.t)

        if self._calibrate_at_each_step:
            # With the updated diffusion, we need to re-compute the covariances of the
            # predicted RV and measured RV.
            # The resulting predicted and measured RV are overwritten herein.
            full_pred_cov_cholesky = utils.linalg.cholesky_update(
                np.sqrt(local_diffusion) * pred_rv_error_free.cov_cholesky, pred_sqrtm
            )
            full_pred_cov = full_pred_cov_cholesky @ full_pred_cov_cholesky.T
            pred_rv = randvars.Normal(
                mean=pred_rv_error_free.mean,
                cov=full_pred_cov,
                cov_cholesky=full_pred_cov_cholesky,
            )

            full_meas_cov_cholesky = utils.linalg.cholesky_update(
                np.sqrt(local_diffusion) * meas_rv_error_free.cov_cholesky, meas_sqrtm
            )
            full_meas_cov = full_meas_cov_cholesky @ full_meas_cov_cholesky.T
            meas_rv = randvars.Normal(
                mean=meas_rv_error_free.mean,
                cov=full_meas_cov,
                cov_cholesky=full_meas_cov_cholesky,
            )

        else:
            # Combine error-free and noisy-component predictions into a full prediction.
            # This has not been assembled as a standalone random variable yet,
            # but is needed for the update below.
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

        # Gain needs manual catching up, too. Use it to compute the update
        crosscov = full_pred_cov @ H.T
        gain = scipy.linalg.cho_solve((meas_rv.cov_cholesky, True), crosscov.T).T
        zero_data = np.zeros(meas_rv.mean.shape)
        filt_rv, _ = self.measurement_model.backward_realization(
            zero_data, pred_rv, rv_forwarded=meas_rv, gain=gain
        )

        t_new = state.t + dt
        state = _odesolver_state.ODESolverState(
            ivp=state.ivp,
            rv=filt_rv,
            t=t_new,
            error_estimate=local_errors,
            reference_state=reference_values,
        )
        return state

    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""

        kalman_posterior = filtsmooth.gaussian.FilteringPosterior(
            transition=self.prior_process.transition,
            locations=times,
            states=rvs,
            diffusion_model=self.diffusion_model,
        )

        return _kalman_odesolution.KalmanODESolution(kalman_posterior)

    def postprocess(self, odesol):
        """If specified (at initialisation), smooth the filter output."""
        locations = odesol.kalman_posterior.locations
        rv_list = odesol.kalman_posterior.states

        kalman_posterior = filtsmooth.gaussian.FilteringPosterior(
            transition=self.prior_process.transition,
            diffusion_model=self.diffusion_model,
        )

        # Constant diffusion model is the only way to go here.
        s = (
            self.diffusion_model.diffusion
            if self._calibrate_all_states_post_hoc
            else 1.0
        )

        for idx, (t, rv) in enumerate(zip(locations, rv_list)):
            kalman_posterior.append(
                location=t,
                state=randvars.Normal(
                    mean=rv.mean,
                    cov=s * rv.cov,
                    cov_cholesky=np.sqrt(s) * rv.cov_cholesky,
                ),
            )

        if self.with_smoothing is True:

            squared_diffusion_list = self.diffusion_model(locations[1:])
            rv_list = self.prior_process.transition.smooth_list(
                rv_list, locations, _diffusion_list=squared_diffusion_list
            )
            kalman_posterior = filtsmooth.gaussian.SmoothingPosterior(
                filtering_posterior=kalman_posterior,
                transition=self.prior_process.transition,
                locations=locations,
                states=rv_list,
                diffusion_model=self.diffusion_model,
            )

        return _kalman_odesolution.KalmanODESolution(kalman_posterior)
