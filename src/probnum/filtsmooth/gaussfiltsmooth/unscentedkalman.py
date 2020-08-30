"""
General Gaussian filters based on approximating intractable
quantities with numerical quadrature.
Examples include the unscented Kalman filter / RTS smoother
which is based on a third degree fully symmetric rule.
"""

import numpy as np

from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import (
    GaussFiltSmooth,
    linear_discrete_update,
)
from probnum.random_variables import Normal
from probnum.filtsmooth.gaussfiltsmooth.unscentedtransform import UnscentedTransform
from probnum.filtsmooth.statespace import (
    ContinuousModel,
    DiscreteModel,
    LinearSDEModel,
    DiscreteGaussianModel,
    DiscreteGaussianLinearModel,
)


class UnscentedKalman(GaussFiltSmooth):
    """
    Factory method for Unscented Kalman filters.
    """

    def __new__(cls, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):

        if cls is UnscentedKalman:
            if _cont_disc(dynamod, measmod):
                return _ContDiscUnscentedKalman(
                    dynamod, measmod, initrv, alpha, beta, kappa, **kwargs
                )
            if _disc_disc(dynamod, measmod):
                return _DiscDiscUnscentedKalman(
                    dynamod, measmod, initrv, alpha, beta, kappa, **kwargs
                )
            else:
                errmsg = (
                    "Cannot instantiate Unscented Kalman filter with "
                    "given dynamic model and measurement model."
                )
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)


def _cont_disc(dynamod, measmod):
    dyna_is_cont = issubclass(type(dynamod), ContinuousModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_cont and meas_is_disc


def _disc_disc(dynamod, measmod):
    dyna_is_disc = issubclass(type(dynamod), DiscreteModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_disc and meas_is_disc


class _ContDiscUnscentedKalman(UnscentedKalman):
    def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError(
                "This implementation of "
                "_ContDiscUnscentedKalman "
                "requires a linear dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError(
                "_DiscDiscUnscentedKalman requires " "a Gaussian measurement model."
            )
        if "cke_nsteps" in kwargs.keys():
            self.cke_nsteps = kwargs["cke_nsteps"]
        else:
            self.cke_nsteps = 1
        super().__init__(dynamod, measmod, initrv)
        self.ut = UnscentedTransform(self.dynamod.ndim, alpha, beta, kappa)

    def predict(self, start, stop, randvar, **kwargs):
        step = (stop - start) / self.cke_nsteps
        return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar, **kwargs)

    def update(self, time, randvar, data, **kwargs):
        return _discrete_unskalman_update(
            time, randvar, data, self.measmod, self.ut, **kwargs
        )


class _DiscDiscUnscentedKalman(UnscentedKalman):
    def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
        """
        Checks that dynamod and measmod are linear and moves on.
        """
        if not issubclass(type(dynamod), DiscreteGaussianModel):
            raise ValueError(
                "_DiscDiscUnscentedKalman requires " "a Gaussian dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError(
                "_DiscDiscUnscentedKalman requires " "a Gaussian measurement model."
            )
        super().__init__(dynamod, measmod, initrv)
        self.ut = UnscentedTransform(self.dynamod.ndim, alpha, beta, kappa)

    def predict(self, start, stop, randvar, **kwargs):
        if issubclass(type(self.dynamod), DiscreteGaussianLinearModel):
            return self._predict_linear(start, randvar, **kwargs)
        else:
            return self._predict_nonlinear(start, randvar, **kwargs)

    def _predict_linear(self, start, randvar, **kwargs):
        """
        Basic Kalman update because model is linear.
        """
        mean, covar = randvar.mean, randvar.cov
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        dynamat = self.dynamod.dynamicsmatrix(start, **kwargs)
        forcevec = self.dynamod.force(start, **kwargs)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        mpred = dynamat @ mean + forcevec
        crosscov = covar @ dynamat.T
        cpred = dynamat @ crosscov + diffmat
        return Normal(mpred, cpred), crosscov

    def _predict_nonlinear(self, start, randvar, **kwargs):
        """
        Executes unscented transform!
        """
        mean, covar = randvar.mean, randvar.cov
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        sigmapts = self.ut.sigma_points(mean, covar)
        proppts = self.ut.propagate(start, sigmapts, self.dynamod.dynamics)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        mpred, cpred, crosscov = self.ut.estimate_statistics(
            proppts, sigmapts, diffmat, mean
        )
        return Normal(mpred, cpred), crosscov

    def update(self, time, randvar, data, **kwargs):
        return _discrete_unskalman_update(
            time, randvar, data, self.measmod, self.ut, **kwargs
        )


def _discrete_unskalman_update(time, randvar, data, measmod, ut, **kwargs):
    if issubclass(type(measmod), DiscreteGaussianLinearModel):
        return _update_discrete_linear(time, randvar, data, measmod, **kwargs)
    else:
        return _update_discrete_nonlinear(time, randvar, data, measmod, ut, **kwargs)


def _update_discrete_linear(time, randvar, data, measmod, **kwargs):
    mpred, cpred = randvar.mean, randvar.cov
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    measmat = measmod.dynamicsmatrix(time, **kwargs)
    meascov = measmod.diffusionmatrix(time, **kwargs)
    meanest = measmat @ mpred
    return linear_discrete_update(meanest, cpred, data, meascov, measmat, mpred)


def _update_discrete_nonlinear(time, randvar, data, measmod, ut, **kwargs):
    mpred, cpred = randvar.mean, randvar.cov
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    sigmapts = ut.sigma_points(mpred, cpred)
    proppts = ut.propagate(time, sigmapts, measmod.dynamics)
    meascov = measmod.diffusionmatrix(time, **kwargs)
    meanest, covest, ccest = ut.estimate_statistics(proppts, sigmapts, meascov, mpred)
    mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
    cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
    return Normal(mean, cov), covest, ccest, meanest
