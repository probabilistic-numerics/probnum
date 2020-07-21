"""
Gaussian filtering and smoothing based on making intractable quantities
tractable through Taylor-method approximations, e.g. linearization.
"""
import numpy as np

from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import *
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace import *


class ExtendedKalman(GaussFiltSmooth):
    """
    Factory method for Kalman filters.
    """

    def __new__(cls, dynamod, measmod, initrv, **kwargs):
        """ """
        if cls is ExtendedKalman:
            if _cont_disc(dynamod, measmod):
                return _ContDiscExtendedKalman(dynamod, measmod, initrv, **kwargs)
            if _disc_disc(dynamod, measmod):
                return _DiscDiscExtendedKalman(dynamod, measmod, initrv, **kwargs)
            else:
                errmsg = (
                    "Cannot instantiate Extended Kalman filter with "
                    "given dynamic model and measurement model."
                )
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)


def _cont_disc(dynamod, measmod):
    """Check whether the state space model is continuous-discrete."""
    dyna_is_cont = issubclass(type(dynamod), ContinuousModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_cont and meas_is_disc


def _disc_disc(dynamod, measmod):
    """Check whether the state space model is discrete-discrete."""
    dyna_is_disc = issubclass(type(dynamod), DiscreteModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_disc and meas_is_disc


class _ContDiscExtendedKalman(ExtendedKalman):
    """
    Continuous-discrete extended Kalman filtering and smoothing.
    """

    def __init__(self, dynamod, measmod, initrv, **kwargs):
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError(
                "This implementation of ContDiscExtendedKalmanFilter "
                "requires a linear dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError(
                "ContDiscExtendedKalmanFilter requires a Gaussian measurement model."
            )
        if "cke_nsteps" in kwargs.keys():
            self.cke_nsteps = kwargs["cke_nsteps"]
        else:
            self.cke_nsteps = 1
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        step = (stop - start) / self.cke_nsteps
        return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar, **kwargs)

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_extkalman_update(
            time, randvar, data, self.measurementmodel, **kwargs
        )


class _DiscDiscExtendedKalman(ExtendedKalman):
    """
    """

    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """
        Checks that dynamod and measmod are linear and moves on.
        """
        if not issubclass(type(dynamod), DiscreteGaussianModel):
            raise ValueError(
                "DiscDiscExtendedKalmanFilter requires " "a linear dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError(
                "DiscDiscExtendedKalmanFilter requires " "a linear measurement model."
            )
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        jacob = self.dynamod.jacobian(start, mean, **kwargs)
        mpred = self.dynamod.dynamics(start, mean, **kwargs)
        crosscov = covar @ jacob.T
        cpred = jacob @ crosscov + diffmat
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_extkalman_update(
            time, randvar, data, self.measurementmodel, **kwargs
        )


def _discrete_extkalman_update(time, randvar, data, measmod, **kwargs):
    """
    """
    mpred, cpred = randvar.mean(), randvar.cov()
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    jacob = measmod.jacobian(time, mpred, **kwargs)
    meascov = measmod.diffusionmatrix(time, **kwargs)
    meanest = measmod.dynamics(time, mpred, **kwargs)
    covest = jacob @ cpred @ jacob.T + meascov
    ccest = cpred @ jacob.T
    mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
    cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
    updated_rv = RandomVariable(distribution=Normal(mean, cov))
    return updated_rv, covest, ccest, meanest
