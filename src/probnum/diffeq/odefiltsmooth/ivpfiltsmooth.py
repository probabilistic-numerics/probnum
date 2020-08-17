import warnings
import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.diffeq import odesolver
from probnum.diffeq.odefiltsmooth.prior import ODEPrior
from probnum.filtsmooth import *
from probnum.diffeq.odesolution import ODESolution


class GaussianIVPFilter(odesolver.ODESolver):
    """
    ODE solver that behaves like a Gaussian filter.


    This is based on continuous-discrete Gaussian filtering.

    Note: this is specific for IVPs and does not apply without
    further considerations to, e.g., BVPs.
    """

    def __init__(self, ivp, gaussfilt, steprl):
        """
        steprule : stepsize rule
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
        self.ivp = ivp
        self.gfilt = gaussfilt
        odesolver.ODESolver.__init__(self, steprl)

    def solve(self, firststep, **kwargs):
        """
        Solve IVP and calibrates uncertainty according
        to Proposition 4 in Tronarp et al.

        Parameters
        ----------
        firststep : float
            First step for adaptive step size rule.
        """
        current_rv = self.gfilt.initialrandomvariable
        t = self.ivp.t0
        times = [t]
        rvs = [current_rv]
        step = firststep
        ssqest, num_steps = 0.0, 0

        while t < self.ivp.tmax:

            t_new = t + step
            pred_rv, _ = self.gfilt.predict(t, t_new, current_rv, **kwargs)

            zero_data = 0.0
            filt_rv, meas_cov, crosscov, meas_mean = self.gfilt.update(
                t_new, pred_rv, zero_data, **kwargs
            )

            errorest, ssq = self._estimate_error(
                filt_rv.mean(), crosscov, meas_cov, meas_mean
            )

            if self.steprule.is_accepted(step, errorest):
                times.append(t_new)
                rvs.append(filt_rv)
                num_steps += 1
                ssqest = ssqest + (ssq - ssqest) / num_steps
                current_rv = filt_rv
                t = t_new

            step = self._suggest_step(step, errorest)
            step = min(step, self.ivp.tmax - t)

        rvs = [
            RandomVariable(distribution=Normal(rv.mean(), ssqest * rv.cov()))
            for rv in rvs
        ]

        return ODESolution(times, rvs, self)

    def odesmooth(self, filter_solution, **kwargs):
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
        ivp_filter_posterior = filter_solution._kalman_posterior
        ivp_smoother_posterior = self.gfilt.smooth(ivp_filter_posterior, **kwargs)

        smoothed_solution = ODESolution(
            times=ivp_smoother_posterior.locations,
            rvs=ivp_smoother_posterior.state_rvs,
            solver=filter_solution._solver,
        )

        return smoothed_solution

    def undo_preconditioning(self, rv):
        ipre = self.gfilt.dynamicmodel.invprecond
        newmean = ipre @ rv.mean()
        newcov = ipre @ rv.cov() @ ipre.T
        newrv = RandomVariable(distribution=Normal(newmean, newcov))
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

    def _suggest_step(self, step, errorest):
        """
        Suggests step according to steprule and warns if
        step is extremely small.

        Raises
        ------
        RuntimeWarning
            If suggested step is smaller than :math:`10^{-15}`.
        """
        step = self.steprule.suggest(step, errorest)
        if step < 1e-15:
            warnmsg = "Stepsize is num. zero (%.1e)" % step
            warnings.warn(message=warnmsg, category=RuntimeWarning)
        return step

    @property
    def prior(self):
        return self.gfilt.dynamicmodel
