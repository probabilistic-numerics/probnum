import numpy as np

from probnum.diffeq import odesolver
from probnum.diffeq.odefiltsmooth.prior import ODEPrior
from probnum.diffeq.odesolution import ODESolution
from probnum.random_variables import Normal


class GaussianIVPFilter(odesolver.ODESolver):
    """ODE solver that behaves like a Gaussian filter.

    This is based on continuous-discrete Gaussian filtering.

    Note: this is specific for IVPs and does not apply without
    further considerations to, e.g., BVPs.

    Parameters
    ----------
    gaussfilt : gaussianfilter.GaussianFilter
        e.g. the return value of ivp_to_ukf(), ivp_to_ekf1().

    Notes
    -----
    - gaussfilt.dynamicmodel contains the prior,
    - gaussfilt.measurementmodel contains the information about the ODE right hand side function,
    - gaussfilt.initialdistribution contains the information about the initial values.
    """

    def __init__(self, ivp, gaussfilt, with_smoothing):
        if not issubclass(type(gaussfilt.dynamicmodel), ODEPrior):
            raise ValueError("Please initialise a Gaussian filter with an ODEPrior")
        self.gfilt = gaussfilt
        self.sigma_squared_mle = 1.0
        self.with_smoothing = with_smoothing
        super().__init__(ivp)

    def initialise(self):
        return self.ivp.t0, self.gfilt.initialrandomvariable

    def step(self, t, t_new, current_rv, **kwargs):
        """Gaussian IVP filter step as nonlinear Kalman filtering with zero data."""
        # 0. Obtain the diffusion matrix; required for calibration / error estimation
        discrete_dynamics = self.gfilt.dynamod.discretise(t_new - t)
        diffmat = discrete_dynamics.diffusionmatrix(t_new)

        # 1. Predict
        pred_rv, _ = self.gfilt.predict(t, t_new, current_rv, **kwargs)

        # 2. Measure
        meas_rv, info = self.gfilt.measure(t_new, pred_rv, **kwargs)

        # 3. Estimate the diffusion (sigma squared)
        self.sigma_squared_mle = self._estimate_diffusion(pred_rv, meas_rv)
        # 3.1. Adjust the prediction covariance to include the diffusion
        pred_rv = Normal(
            pred_rv.mean, pred_rv.cov + (self.sigma_squared_mle - 1) * diffmat
        )
        # 3.2 Update the measurement covariance (measure again)
        meas_rv, info = self.gfilt.measure(t_new, pred_rv, **kwargs)

        # 4. Update
        zero_data = 0.0
        filt_rv = self.gfilt.condition_state_on_measurement(
            pred_rv, meas_rv, zero_data, info["crosscov"], **kwargs
        )

        # 5. Error estimate
        local_errors = self._estimate_local_error(
            pred_rv, t_new, self.sigma_squared_mle * diffmat
        )
        err = np.linalg.norm(local_errors)

        return filt_rv, err

    def postprocess(self, times, rvs):
        """Rescale covariances with sigma square estimate, (if specified) smooth the
        estimate, return ODESolution."""
        if False:  # pylint: disable=using-constant-test
            # will become useful again for time-fixed diffusion models
            rvs = self._rescale(rvs)
        odesol = super().postprocess(times, rvs)
        if self.with_smoothing is True:
            odesol = self._odesmooth(ode_solution=odesol)
        return odesol

    def _rescale(self, rvs):
        """Rescales covariances according to estimate sigma squared value."""
        rvs = [Normal(rv.mean, self.sigma_squared_mle * rv.cov) for rv in rvs]
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
        ivp_filter_posterior = ode_solution._kalman_posterior
        ivp_smoother_posterior = self.gfilt.smooth(ivp_filter_posterior, **kwargs)

        smoothed_solution = ODESolution(
            times=ivp_smoother_posterior.locations,
            rvs=ivp_smoother_posterior.state_rvs,
            solver=ode_solution._solver,
        )

        return smoothed_solution

    def undo_preconditioning(self, rv):
        ipre = self.gfilt.dynamicmodel.invprecond
        newmean = ipre @ rv.mean
        newcov = ipre @ rv.cov @ ipre.T
        newrv = Normal(newmean, newcov)
        return newrv

    def _estimate_local_error(self, pred_rv, t_new, calibrated_diffmat, **kwargs):
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
        local_pred_rv = Normal(pred_rv.mean, calibrated_diffmat)
        local_meas_rv, _ = self.gfilt.measure(t_new, local_pred_rv, **kwargs)
        error = local_meas_rv.cov.diagonal()
        return error

    def _estimate_diffusion(self, pred_rv, meas_rv):
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
        std_like = np.linalg.cholesky(meas_rv.cov)
        whitened_res = np.linalg.solve(std_like, meas_rv.mean)
        ssq = whitened_res @ whitened_res / meas_rv.size
        return ssq

    @property
    def prior(self):
        return self.gfilt.dynamicmodel
