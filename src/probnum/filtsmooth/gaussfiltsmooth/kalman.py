"""
Kalman filtering and (Rauch-Tung-Striebel) smoothing for
continuous-discrete and discrete-discrete state space models.
"""

import numpy as np
from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import (
    GaussFiltSmooth,
    linear_discrete_update,
)
from probnum.random_variables import Normal
from probnum.filtsmooth.statespace import LinearSDEModel, DiscreteGaussianLinearModel

from probnum.filtsmooth.gaussfiltsmooth._utils import is_cont_disc, is_disc_disc


class Kalman(GaussFiltSmooth):
    """
    Kalman filtering and smoothing for continuous-discrete and
    discrete-discrete state space models.
    """

    def __new__(cls, dynamod, measmod, initrv, **kwargs):
        """
        Factory method for Kalman filtering and smoothing.

        Depending on whether the dynamic model is continuous or
        discrete, either a continuous-discrete Kalman object or a
        discrete-discrete Kalman object is created.
        """
        if cls is Kalman:
            if is_cont_disc(dynamod, measmod):
                return _ContDiscKalman(dynamod, measmod, initrv, **kwargs)
            if is_disc_disc(dynamod, measmod):
                return _DiscDiscKalman(dynamod, measmod, initrv)
            else:
                errmsg = (
                    "Cannot instantiate Kalman object with given "
                    "dynamic model and measurement model."
                )
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)


class _ContDiscKalman(Kalman):
    """
    Provides predict() and update() methods for Kalman filtering and
    smoothing on continuous-discrete state space models.
    """

    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """
        Checks that dynamod and measmod are linear and moves on.
        """
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError(
                "ContinuousDiscreteKalman requires a linear dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianLinearModel):
            raise ValueError(
                "ContinuousDiscreteKalman requires a linear measurement model."
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
        return _discrete_kalman_update(
            time, randvar, data, self.measurementmodel, **kwargs
        )


class _DiscDiscKalman(Kalman):
    """
    Provides predict() and update() methods for Kalman filtering and
    smoothing on discrete-discrete state space models.
    """

    def __init__(self, dynamod, measmod, initrv):
        """Checks that dynamod and measmod are linear and moves on."""
        if not issubclass(type(dynamod), DiscreteGaussianLinearModel):
            raise ValueError(
                "ContinuousDiscreteKalman requires " "a linear dynamic model."
            )
        if not issubclass(type(measmod), DiscreteGaussianLinearModel):
            raise ValueError(
                "DiscreteDiscreteKalman requires " "a linear measurement model."
            )
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        """Prediction step for discrete-discrete Kalman filtering."""
        mean, covar = randvar.mean, randvar.cov
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        dynamat = self.dynamicmodel.dynamicsmatrix(start, **kwargs)
        forcevec = self.dynamicmodel.force(start, **kwargs)
        diffmat = self.dynamicmodel.diffusionmatrix(start, **kwargs)
        mpred = dynamat @ mean + forcevec
        ccpred = covar @ dynamat.T
        cpred = dynamat @ ccpred + diffmat
        return Normal(mpred, cpred), ccpred

    def update(self, time, randvar, data, **kwargs):
        """Update step of discrete Kalman filtering"""
        return _discrete_kalman_update(
            time, randvar, data, self.measurementmodel, **kwargs
        )


def _discrete_kalman_update(time, randvar, data, measurementmodel, **kwargs):
    """Discrete Kalman update."""
    mpred, cpred = randvar.mean, randvar.cov
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    measmat = measurementmodel.dynamicsmatrix(time, **kwargs)
    meascov = measurementmodel.diffusionmatrix(time, **kwargs)
    meanest = measmat @ mpred
    return linear_discrete_update(meanest, cpred, data, meascov, measmat, mpred)
