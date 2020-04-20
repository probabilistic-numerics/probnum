"""
Check if all filters have the same solve() method.
If yes, make a ODEFilter() object.
If not, make ODEKalmanFilter, ODEExtendedKalmanFilter, ... objects.

Desired
-------
diffeq.solver.solve_filter(ode, stepsize=0.1, prior='ibm_4', filter="kalman")
diffeq.solver.solve_filter(ode, tol=1e-04, prior='matern_52', filter="ekf1")


Beware
------
Unittests for this function are yet to be transferred.
"""

import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.diffeq import odesolver
from probnum.filtsmooth import GaussianSmoother


class GaussianIVPSmoother(odesolver.ODESolver):
    """
    ODE solver that behaves like a Gaussian smoother.

    Builds on top of Gaussian IVP Filter.
    """
    def __init__(self, ivp, gaussfilt, steprl):
        """ """
        self.gauss_ode_filt = GaussianIVPFilter(ivp, gaussfilt, steprl)
        self.smoother = GaussianSmoother(gaussfilt)

    def solve(self, firststep, **kwargs):
        """
        """
        means, covars, times = self.gauss_ode_filt.solve(firststep, **kwargs)
        smoothed_means, smoothed_covars = self.smoother.smooth_filterout(means, covars, times, **kwargs)
        return smoothed_means, smoothed_covars, times


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
        self.ivp = ivp
        self.gfilt = gaussfilt
        odesolver.ODESolver.__init__(self, steprl)

    def solve(self, firststep, **kwargs):
        """
        Solves IVP and calibrates uncertainty according
        to Proposition 4 in Tronarp et al.

        Parameters
        ----------
        firststep : float
            First step for adaptive step size rule.
        """
        current = self.gfilt.initialdistribution
        step = firststep
        ssqest, ct = 0.0, 0
        times, means, covars = [self.ivp.t0], [current.mean()], [current.cov()]
        if "nsteps" in kwargs.keys():
            nsteps = kwargs["nsteps"]
        else:
            nsteps = 1
        while times[-1] < self.ivp.tmax:
            intermediate_step = float(step / nsteps)
            tm = times[-1]
            interms, intercs, interts = [], [], []
            for idx in range(nsteps):
                newtm = tm + intermediate_step
                current, __ = self.gfilt.predict(tm, newtm, current, **kwargs)
                interms.append(current.mean())
                intercs.append(current.cov())
                interts.append(newtm)
                tm = newtm
            predicted = current
            new_time = tm
            # predicted, __ = self.gfilt.predict(times[-1], new_time, current, **kwargs)
            zero_data = 0.0
            current, covest, ccest, mnest = self.gfilt.update(new_time, predicted, zero_data, **kwargs)
            interms[-1] = current.mean()
            intercs[-1] = current.cov()
            errorest, ssq = self._estimate_error(current.mean(), ccest, covest, mnest)
            if self.steprule.is_accepted(step, errorest) is True:
                times.extend(interts)
                means.extend(interms)
                covars.extend(intercs)
                ct = ct + 1
                ssqest = (ssqest + (ssq - ssqest) / ct)
            else:
                current = RandomVariable(distribution=Normal(means[-1], covars[-1]))
            step = self._suggest_step(step, errorest)
        return np.array(means), ssqest * np.array(covars), np.array(times)

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
        ssq = whitened_res @ whitened_res
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
        rel_error = (abserrors / np.abs(projmat @ currmn)) @ weights / np.linalg.norm(weights)
        abs_error = abserrors @ weights / np.linalg.norm(weights)
        return np.maximum(rel_error, abs_error)



    def _suggest_step(self, step, errorest):
        """
        Suggests step according to steprule and warns if
        step is extremely small.
        """
        step = self.steprule.suggest(step, errorest)
        if step < 1e-15:
            print("Warning: Stepsize is num. zero (%.1e)" % step)
        return step




