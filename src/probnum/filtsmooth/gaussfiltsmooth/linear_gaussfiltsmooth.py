"""
Kalman filtering and (Rauch-Tung-Striebel) smoothing for
continuous-discrete and discrete-discrete state space models.
"""
import numpy as np

from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import *
from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth.statespace import *


class RauchTungStriebelSmoother(GaussianSmoother):
    """
    Rauch-Tung-Striebel smoother.

    Gaussian smoother based on Kalman filter instances.
    """

    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """
        Makes a KalmanFilter instance and the rest ist taken over
        by the superclass.
        """
        kalfilt = KalmanFilter(dynamod, measmod, initrv, **kwargs)
        super().__init__(kalfilt)


class KalmanFilter:
    """
    Factory method for Kalman filters.
    """
    def __new__(cls, dynamod, measmod, initrv, **kwargs):
        """ """
        if cls is KalmanFilter:
            if _cont_disc(dynamod, measmod):
                return ContDiscKalmanFilter(dynamod, measmod, initrv, **kwargs)
            if _disc_disc(dynamod, measmod):
                return DiscDiscKalmanFilter(dynamod, measmod, initrv)
            else:
                errmsg = ("Cannot instantiate Kalman filter with given "
                          "dynamic model and measurement model.")
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)


def _cont_disc(dynamod, measmod):
    """ """
    dyna_is_cont = issubclass(type(dynamod), ContinuousModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_cont and meas_is_disc


def _disc_disc(dynamod, measmod):
    """ """
    dyna_is_disc = issubclass(type(dynamod), DiscreteModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_disc and meas_is_disc


class ContDiscKalmanFilter(ContDiscGaussianFilter, KalmanFilter):
    """
    Completes implementation of ContinuousContinuousGaussianFilter.

    Provides predict() and update() methods.
    """
    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """
        Checks that dynamod and measmod are linear and moves on.
        """
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError("ContinuosDiscreteKalmanFilter requires "
                             "a linear dynamic model.")
        if not issubclass(type(measmod), DiscreteGaussianLinearModel):
            raise ValueError("DiscreteDiscreteKalmanFilter requires "
                             "a linear measurement model.")
        if "cke_nsteps" in kwargs.keys():
            self.cke_nsteps = kwargs["cke_nsteps"]
        else:
            self.cke_nsteps = 1
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        step = ((stop - start) / self.cke_nsteps)
        return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar,
                                                   **kwargs)

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_kalman_update(time, randvar, data,
                                       self.measurementmodel, **kwargs)


class DiscDiscKalmanFilter(DiscDiscGaussianFilter, KalmanFilter):
    """
    """
    def __init__(self, dynamod, measmod, initrv):
        """
        Checks that dynamod and measmod are linear and moves on.
        """
        if not issubclass(type(dynamod), DiscreteGaussianLinearModel):
            raise ValueError("ContinuousDiscreteKalmanFilter requires "
                             "a linear dynamic model.")
        if not issubclass(type(measmod), DiscreteGaussianLinearModel):
            raise ValueError("DiscreteDiscreteKalmanFilter requires "
                             "a linear measurement model.")
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean * np.ones(1), covar * np.eye(1)
        dynamat = self.dynamicmodel.dynamicsmatrix(start, **kwargs)
        forcevec = self.dynamicmodel.force(start, **kwargs)
        diffmat = self.dynamicmodel.diffusionmatrix(start, **kwargs)
        mpred = dynamat @ mean + forcevec
        ccpred = covar @ dynamat.T
        cpred = dynamat @ ccpred + diffmat
        return RandomVariable(distribution=Normal(mpred, cpred)), ccpred

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_kalman_update(time, randvar, data,
                                       self.measurementmodel, **kwargs)


def _discrete_kalman_update(time, randvar, data, measurementmodel, **kwargs):
    """
    """
    mpred, cpred = randvar.mean(), randvar.cov()
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    measmat = measurementmodel.dynamicsmatrix(time, **kwargs)
    meascov = measurementmodel.diffusionmatrix(time, **kwargs)
    meanest = measmat @ mpred
    covest = measmat @ cpred @ measmat.T + meascov
    ccest = cpred @ measmat.T
    mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
    cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
    return RandomVariable(distribution=Normal(mean, cov)), \
        covest, ccest, meanest
