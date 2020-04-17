"""
Check if all filters have the same solve() method.
If yes, make a ODEFilter() object.
If not, make ODEKalmanFilter, ODEExtendedKalmanFilter, ... objects.

Desired
-------
diffeq.solver.solve_filter(ode, stepsize=0.1, prior='ibm_4', filter="kalman")
diffeq.solver.solve_filter(ode, tol=1e-04, prior='matern_52', filter="ekf")


Beware
------
Unittests for this function are yet to be transferred.
"""

import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.diffeq import odesolver, steprule
from probnum.diffeq.odefilter import prior, ivptofilter


class GaussianIVPFilter(odesolver.ODESolver):
    """
    ODE solver that behaves like a Gaussian filter.

    Note: this is specific for IVPs and does not apply without
    further considerations to, e.g., BVPs.
    """

    def __init__(self, ivp, gaussfilt, steprl):
        """
        steprule : stepsize rule
        gaussfilt : gaussianfilter.GaussianFilter object,
            e.g. the return value of ivp_to_ukf(), ivp_to_ekf().

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
        while times[-1] < self.ivp.tmax:
            new_time = times[-1] + step
            predicted, __ = self.gfilt.predict(times[-1], new_time, current, **kwargs)
            zero_data = 0.0
            current, covest, ccest, mnest = self.gfilt.update(new_time, predicted, zero_data, **kwargs)
            errorest, ssq = self._estimate_error(current.mean(), ccest, covest, mnest)
            if self.steprule.is_accepted(step, errorest) is True:
                times.append(new_time)
                means.append(current.mean())
                covars.append(current.cov())
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



def filter_ivp(ivp, tol, which_prior="ibm1", which_filt="kf", firststep=None, **pars):
    """
    Solve ivp with adaptive step size.

    Easy way out. No option to choose interesting priors
    (with parameters). For better tuning, use the objects.

    Turns prior-string into actual prior,
    filt-string into actual filter,
    creats a GaussianODEFilter object and calls solve().

    which_prior : string, element of
        [ibm1, ibm2, ibm3,
         ioup1, ioup2, ioup3,
         matern32, matern52, matern72]

    which_filter : string, element of [kf, ekf, ukf]
    step : float
    ivp : IVP object
    """
    prior = _string_to_prior(ivp, which_prior, **pars)
    gfilt = _string_to_filter(ivp, prior, which_filt, **pars)
    stprl = _step_to_adaptive_steprule(tol, prior)
    ofi = GaussianIVPFilter(ivp, gfilt, stprl)
    if firststep == None:
        firststep = ivp.tmax - ivp.t0
    return ofi.solve(firststep)


def filter_ivp_h(ivp, step, which_prior="ibm1", which_filt="kf", **pars):
    """
    Solve ivp with constant step size.

    Easy way out. No option to choose interesting priors
    (with parameters). For better tuning, use the objects.

    Turns prior-string into actual prior,
    filt-string into actual filter,
    creats a GaussianODEFilter object and calls solve().

    which_prior : string, element of
        [ibm1, ibm2, ibm3, ibm4, ibm5,
         ioup1, ioup2, ioup3, ioup4, ioup5,
         matern32, matern52, matern72, matern92]

    which_prior : string, element of {kf, ekf, ukf}
    step : float
    ivp : IVP object
    """
    prior = _string_to_prior(ivp, which_prior, **pars)
    gfilt = _string_to_filter(ivp, prior, which_filt, **pars)
    stprl = _step_to_steprule(step)
    ofi = GaussianIVPFilter(ivp, gfilt, stprl)
    return ofi.solve(firststep=step)

def _string_to_prior(ivp, which_prior, **pars):
    """
    """
    ibm_family = ["ibm1", "ibm2", "ibm3"]
    ioup_family = ["ioup1", "ioup2", "ioup3"]
    matern_family = ["matern32", "matern52", "matern72"]
    if which_prior in ibm_family:
        return _string_to_prior_ibm(ivp, which_prior, **pars)
    elif which_prior in ioup_family:
        return _string_to_prior_ioup(ivp, which_prior, **pars)
    elif which_prior in matern_family:
        return _string_to_prior_matern(ivp, which_prior, **pars)
    else:
        raise TypeError("Type of prior not supported.")

def _string_to_prior_ibm(ivp, which_prior, **pars):
    """
    """
    if "diffconst" in pars.keys():
        diffconst = pars["diffconst"]
    else:
        diffconst = 1.0
    if which_prior == "ibm1":
        return prior.IBM(1, ivp.ndim, diffconst)
    elif which_prior == "ibm2":
        return prior.IBM(2, ivp.ndim, diffconst)
    elif which_prior == "ibm3":
        return prior.IBM(3, ivp.ndim, diffconst)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")

def _string_to_prior_ioup(ivp, which_prior, **pars):
    """
    """
    if "diffconst" in pars.keys():
        diffconst = pars["diffconst"]
    else:
        diffconst = 1.0
    if "driftspeed" in pars.keys():
        driftspeed = pars["driftspeed"]
    else:
        driftspeed = 1.0
    if which_prior == "ioup1":
        return prior.IOUP(1, ivp.ndim, driftspeed, diffconst)
    elif which_prior == "ioup2":
        return prior.IOUP(2, ivp.ndim, driftspeed, diffconst)
    elif which_prior == "ioup3":
        return prior.IOUP(3, ivp.ndim, driftspeed, diffconst)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")

def _string_to_prior_matern(ivp, which_prior, **pars):
    """
    """
    if "diffconst" in pars.keys():
        diffconst = pars["diffconst"]
    else:
        diffconst = 1.0
    if "lengthscale" in pars.keys():
        lengthscale = pars["lengthscale"]
    else:
        lengthscale = 1.0
    if which_prior == "matern32":
        return prior.Matern(1, ivp.ndim, lengthscale, diffconst)
    elif which_prior == "matern52":
        return prior.Matern(2, ivp.ndim, lengthscale, diffconst)
    elif which_prior == "matern72":
        return prior.Matern(3, ivp.ndim, lengthscale, diffconst)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string_to_filter(ivp, prior, which_filt, **pars):
    """
    """
    if "evlvar" in pars.keys():
        evlvar = pars["evlvar"]
    else:
        evlvar = 0.0
    if which_filt == "kf":
        return ivptofilter.ivp_to_kf(ivp, prior, evlvar)
    elif which_filt == "ekf":
        return ivptofilter.ivp_to_ekf(ivp, prior, evlvar)
    elif which_filt == "ukf":
        return ivptofilter.ivp_to_ukf(ivp, prior, evlvar)
    else:
        raise TypeError("Type of filter not supported.")

def _step_to_steprule(stp):
    """
    """
    return steprule.ConstantSteps(stp)


def _step_to_adaptive_steprule(tol, prior, **pars):
    """
    """
    convrate = prior.ordint + 1
    return steprule.AdaptiveSteps(tol, convrate, **pars)

