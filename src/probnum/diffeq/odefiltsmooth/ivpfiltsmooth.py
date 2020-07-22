"""
"""
import warnings
import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.diffeq import odesolver
from probnum.diffeq.odefiltsmooth.prior import ODEPrior
from probnum.filtsmooth import *


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

    def solve(self, firststep, nsteps=1, **kwargs):
        """
        Solves IVP and calibrates uncertainty according
        to Proposition 4 in Tronarp et al.

        Parameters
        ----------
        firststep : float
            First step for adaptive step size rule.
        nsteps : int, optional
            Number of inbetween steps for the filter. Default is 1.
        """

        ####### This function surely can use some code cleanup. #######

        current = self.gfilt.initialdistribution
        step = firststep
        ssqest, ct = 0.0, 0
        times, means, covars = [self.ivp.t0], [current.mean()], [current.cov()]
        while times[-1] < self.ivp.tmax:
            intermediate_step = float(step / nsteps)
            tm = times[-1]
            interms, intercs, interts = [], [], []
            for idx in range(nsteps):
                newtm = tm + intermediate_step
                current, __ = self.gfilt.predict(tm, newtm, current, **kwargs)
                interms.append(current.mean().copy())
                intercs.append(current.cov().copy())
                interts.append(newtm)
                tm = newtm
            predicted = current
            new_time = tm
            zero_data = 0.0
            current, covest, ccest, mnest = self.gfilt.update(
                new_time, predicted, zero_data, **kwargs
            )
            interms[-1] = current.mean().copy()
            intercs[-1] = current.cov().copy()
            errorest, ssq = self._estimate_error(current.mean(), ccest, covest, mnest)
            if self.steprule.is_accepted(step, errorest) is True:
                times.extend(interts)
                means.extend(interms)
                covars.extend(intercs)
                ct = ct + 1
                ssqest = ssqest + (ssq - ssqest) / ct
            else:
                current = RandomVariable(distribution=Normal(means[-1], covars[-1]))
            step = self._suggest_step(step, errorest)
        means, covars = self.undo_preconditioning(means, covars)
        return np.array(means), ssqest * np.array(covars), np.array(times)

    def odesmooth(self, means, covs, times, **kwargs):
        """
        Smoothes out the ODE-Filter output.

        Be careful about the preconditioning: the GaussFiltSmooth object
        only knows the state space with changed coordinates!

        Parameters
        ----------
        means
        covs

        Returns
        -------

        """
        means, covs = self.redo_preconditioning(means, covs)
        means, covs = self.gfilt.smooth(means, covs, times, **kwargs)
        return self.undo_preconditioning(means, covs)

    def undo_preconditioning(self, means, covs):
        """ """
        ipre = self.gfilt.dynamicmodel.invprecond
        newmeans = np.array([ipre @ mean for mean in means])
        newcovs = np.array([ipre @ cov @ ipre.T for cov in covs])
        return newmeans, newcovs

    def redo_preconditioning(self, means, covs):
        """ """
        pre = self.gfilt.dynamicmodel.precond
        newmeans = np.array([pre @ mean for mean in means])
        newcovs = np.array([pre @ cov @ pre.T for cov in covs])
        return newmeans, newcovs

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
        """ """
        return self.gfilt.dynamicmodel
