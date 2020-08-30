"""
Gaussian filtering and smoothing based on making intractable quantities
tractable through Taylor-method approximations, e.g. linearization.
"""
import numpy as np

from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import (
    GaussFiltSmooth,
    linear_discrete_update,
)
from probnum.random_variables import Normal
from probnum.filtsmooth.statespace import LinearSDEModel, DiscreteGaussianModel

from probnum.filtsmooth.gaussfiltsmooth._utils import is_cont_disc, is_disc_disc


class ExtendedKalman(GaussFiltSmooth):
    """
    Factory method for Kalman filters.
    """

    def __new__(cls, dynamod, measmod, initrv, **kwargs):

        if cls is ExtendedKalman:
            if is_cont_disc(dynamod, measmod):
                return _ContDiscExtendedKalman(dynamod, measmod, initrv, **kwargs)
            if is_disc_disc(dynamod, measmod):
                return _DiscDiscExtendedKalman(dynamod, measmod, initrv, **kwargs)
            else:
                errmsg = (
                    "Cannot instantiate Extended Kalman filter with "
                    "given dynamic model and measurement model."
                )
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)


class _ContDiscExtendedKalman(ExtendedKalman):
    """
    Continuous-discrete extended Kalman filtering and smoothing.
    """

    def __init__(self, dynamod, measmod, initrv, **kwargs):
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError(
                "This implementation of ContDiscExtendedKalman "
                "requires a linear dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError(
                "ContDiscExtendedKalman requires a Gaussian measurement model."
            )
        if "cke_nsteps" in kwargs.keys():
            self.cke_nsteps = kwargs["cke_nsteps"]
        else:
            self.cke_nsteps = 1
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        step = (stop - start) / self.cke_nsteps
        return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar, **kwargs)

    def update(self, time, randvar, data, **kwargs):
        return _discrete_extkalman_update(
            time, randvar, data, self.measurementmodel, **kwargs
        )


class _DiscDiscExtendedKalman(ExtendedKalman):
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
        mean, covar = randvar.mean, randvar.cov
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        jacob = self.dynamod.jacobian(start, mean, **kwargs)
        mpred = self.dynamod.dynamics(start, mean, **kwargs)
        crosscov = covar @ jacob.T
        cpred = jacob @ crosscov + diffmat
        return Normal(mpred, cpred), crosscov

    def update(self, time, randvar, data, **kwargs):
        return _discrete_extkalman_update(
            time, randvar, data, self.measurementmodel, **kwargs
        )


def _discrete_extkalman_update(time, randvar, data, measmod, **kwargs):
    mpred, cpred = randvar.mean, randvar.cov
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    jacob = measmod.jacobian(time, mpred, **kwargs)
    meascov = measmod.diffusionmatrix(time, **kwargs)
    meanest = measmod.dynamics(time, mpred, **kwargs)
    return linear_discrete_update(meanest, cpred, data, meascov, jacob, mpred)
