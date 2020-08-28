import numpy as np

from probnum.random_variables import Normal
from probnum.diffeq import odesolver
from probnum.diffeq.odefiltsmooth.prior import ODEPrior
from probnum.diffeq.odesolution import ODESolution


class GaussianIVPFilter(odesolver.ODESolver):
    """
    ODE solver that behaves like a Gaussian filter.


    This is based on continuous-discrete Gaussian filtering.

    Note: this is specific for IVPs and does not apply without
    further considerations to, e.g., BVPs.
    """

    def __init__(self, ivp, gaussfilt, with_smoothing):
        """
        gaussfilt : gaussianfilter.GaussianFilter object,
            e.g. the return value of ivp_to_ukf(), ivp_to_ekf1().

        Notes
        -----
        * gaussfilt.dynamicmodel contains the prior,
        * gaussfilt.measurementmodel contains the information about the
        ODE right hand side function,
        * gaussfilt.initialdistribution contains the information about
        the initial values.
        """
        if not issubclass(type(gaussfilt.dynamicmodel), ODEPrior):
            raise ValueError("Please initialise a Gaussian filter with an ODEPrior")
        self.gfilt = gaussfilt
        self.sigma_squared_global = 0.0
        self.sigma_squared_current = 0.0
        self.with_smoothing = with_smoothing
        super().__init__(ivp)

    def initialise(self):
        return self.ivp.t0, self.gfilt.initialrandomvariable

    def step(self, t, t_new, current_rv, **kwargs):
        """Gaussian IVP filter step as nonlinear Kalman filtering with zero data."""
        pred_rv, _ = self.gfilt.predict(t, t_new, current_rv, **kwargs)
        zero_data = 0.0
        filt_rv, meas_cov, crosscov, meas_mean = self.gfilt.update(
            t_new, pred_rv, zero_data, **kwargs
        )
        errorest, self.sigma_squared_current = self._estimate_error(
            filt_rv.mean, crosscov, meas_cov, meas_mean
        )
        return filt_rv, errorest

    def method_callback(self, time, current_guess, current_error):
        """Update the sigma-squared (ssq) estimate."""
        self.sigma_squared_global = (
            self.sigma_squared_global
            + (self.sigma_squared_current - self.sigma_squared_global) / self.num_steps
        )

    def postprocess(self, times, rvs):
        """
        Rescale covariances with sigma square estimate,
        (if specified) smooth the estimate, return ODESolution.
        """
        rvs = self._rescale(rvs)
        odesol = super().postprocess(times, rvs)
        if self.with_smoothing is True:
            odesol = self._odesmooth(ode_solution=odesol)
        return odesol

    def _rescale(self, rvs):
        """Rescales covariances according to estimate sigma squared value."""
        rvs = [Normal(rv.mean, self.sigma_squared_global * rv.cov) for rv in rvs]
        return rvs

    def _odesmooth(self, ode_solution, **kwargs):
        """
        Smooth out the ODE-Filter output.

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

    def _estimate_error(self, currmn, ccest, covest, mnest):
        """
        Error estimate.

        Estimates error as whitened residual using the
        residual as weight vector.

        THIS IS NOT A PERFECT ERROR ESTIMATE, but this is a question of
        research, not a question of implementation as of now.
        """
        std_like = np.linalg.cholesky(covest)
        whitened_res = np.linalg.solve(std_like, mnest)
        ssq = whitened_res @ whitened_res / mnest.size
        abserrors = np.abs(whitened_res)
        errorest = self._rel_and_abs_error(abserrors, currmn)
        return errorest, ssq

    def _rel_and_abs_error(self, abserrors, currmn):
        """
        Returns maximum of absolute and relative error.
        This way, both are guaranteed to be satisfied.
        """
        ordint, spatialdim = self.gfilt.dynamicmodel.ordint, self.ivp.ndim
        h0_1d = np.eye(ordint + 1)[:, 0].reshape((1, ordint + 1))
        projmat = np.kron(np.eye(spatialdim), h0_1d)
        weights = np.ones(len(abserrors))
        rel_error = (
            (abserrors / np.abs(projmat @ currmn)) @ weights / np.linalg.norm(weights)
        )
        abs_error = abserrors @ weights / np.linalg.norm(weights)
        return np.maximum(rel_error, abs_error)

    @property
    def prior(self):
        return self.gfilt.dynamicmodel
