"""
"""
import numpy as np

from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import *
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace import *


class ExtendedKalmanSmoother(GaussianSmoother):
    """
    ExtendedKalman smoother as a simple add-on to the filtering.
    The rest is implemented in GaussianSmoother.
    """

    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """ """
        extkalfilt = ExtendedKalmanFilter(dynamod, measmod, initrv, **kwargs)
        super().__init__(extkalfilt)



class ExtendedKalmanFilter:
    """
    Factory method for Kalman filters.
    """
    def __new__(cls, dynamod, measmod, initrv, **kwargs):
        """ """
        if cls is ExtendedKalmanFilter:
            if _cont_cont(dynamod, measmod):
                return ContContExtendedKalmanFilter(dynamod, measmod,
                                                    initrv, **kwargs)
            if _cont_disc(dynamod, measmod):
                return ContDiscExtendedKalmanFilter(dynamod, measmod,
                                                    initrv, **kwargs)
            if _disc_disc(dynamod, measmod):
                return DiscDiscExtendedKalmanFilter(dynamod, measmod,
                                                    initrv, **kwargs)
            else:
                errmsg = ("Cannot instantiate Extended Kalman filter with "
                          "given dynamic model and measurement model.")
                raise ValueError(errmsg)
        else:
            return super().__new__(cls)


def _cont_cont(dynamod, measmod):
    """ """
    dyna_is_cont = issubclass(type(dynamod), ContinuousModel)
    meas_is_cont = issubclass(type(measmod), ContinuousModel)
    return dyna_is_cont and meas_is_cont


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


class ContContExtendedKalmanFilter(ContContGaussianFilter,
                                   ExtendedKalmanFilter):
    """
    Not implemented.

    Error is raised in super().__init__().

    If you'd like to implement it, do your magic here.
    """
    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """ """
        raise NotImplementedError("Continuous-Continuous Extended "
                                  "Kalman Filtering is not implemented")


class ContDiscExtendedKalmanFilter(ContDiscGaussianFilter,
                                   ExtendedKalmanFilter):
    """
    Completes implementation of ContinuousContinuousGaussianFilter.

    Provides predict() and update() methods.
    """
    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """
        """
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError("This implementation of "
                            "ContDiscExtendedKalmanFilter "
                            "requires a linear dynamic model.")
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError("DiscreteDiscreteKalmanFilter requires "
                            "a Gaussian measurement model.")
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
        return _discrete_extkalman_update(time, randvar, data,
                                          self.measurementmodel, **kwargs)


class DiscDiscExtendedKalmanFilter(DiscDiscGaussianFilter, ExtendedKalmanFilter):
    """
    """
    def __init__(self, dynamod, measmod, initrv, **kwargs):
        """
        Checks that dynamod and measmod are linear and moves on.
        """
        if not issubclass(type(dynamod), DiscreteGaussianModel):
            raise ValueError("DiscDiscExtendedKalmanFilter requires "
                            "a linear dynamic model.")
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError("DiscDiscExtendedKalmanFilter requires "
                            "a linear measurement model.")
        super().__init__(dynamod, measmod, initrv)

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean*np.ones(1), covar*np.eye(1)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        jacob = self.dynamod.jacobian(start, mean, **kwargs)
        mpred = self.dynamod.dynamics(start, mean, **kwargs)
        crosscov = covar @ jacob.T
        cpred = jacob @ crosscov + diffmat
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_extkalman_update(time, randvar, data,
                                          self.measurementmodel, **kwargs)


def _discrete_extkalman_update(time, randvar, data, measurementmodel, **kwargs):
    """
    """
    mpred, cpred = randvar.mean(), randvar.cov()
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred * np.ones(1), cpred * np.eye(1)
    jacob = measurementmodel.jacobian(time, mpred, **kwargs)
    meascov = measurementmodel.diffusionmatrix(time, **kwargs)
    meanest = measurementmodel.dynamics(time, mpred, **kwargs)
    covest = jacob @ cpred @ jacob.T + meascov
    ccest = cpred @ jacob.T
    mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
    cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
    return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest




































#
#
#
# class ExtendedKalmanFilter(GaussianFilter):
#     """
#     """
#
#     def __init__(self, dynamod, measmod, initrv, _nsteps=15):
#         """
#         dynmod : continuousmodel.linear.Linear or subclass
#         measmod : discretemodel.gaussmarkov.Nonlinear or subclass
#         initrv : polynomial.MultivariateGaussian
#         _nsteps : approximate integration.
#
#         Functionality so far restricted to linear SDEs because these implement
#         solutions to the CKE. Support of nonlinear equations can be supported
#         through overwriting the "update" step below and hard-coding the solution
#         of the differential equation in Linear.chapmankolmogorov().
#         """
#         if _is_not_linear(dynamod):
#             raise ValueError(
#                 "Extended Kalman filter doesn't support dynamic model")
#         if not issubclass(type(measmod), DiscreteGaussianModel):
#             raise ValueError(
#                 "Extended Kalman filter doesn't support measurement model")
#         if not issubclass(type(initrv.distribution), Normal):
#             raise ValueError("Extended Kalman filter doesn't support "
#                             "initial distribution")
#         self.dynamod = dynamod
#         self.measmod = measmod
#         self.initrv = initrv
#         self._nsteps = _nsteps
#
#     @property
#     def dynamicmodel(self):
#         """ """
#         return self.dynamod
#
#     @property
#     def measurementmodel(self):
#         """ """
#         return self.measmod
#
#     @property
#     def initialdistribution(self):
#         """ """
#         return self.initrv
#
#     def predict(self, start, stop, randvar, **kwargs):
#         """
#         """
#         if _is_discrete(self.dynamod):
#             return self._predict_discrete(start, randvar, **kwargs)
#         else:
#             return self._predict_continuous(start, stop, randvar,
#                                             **kwargs)
#
#     def _predict_discrete(self, start, randvar, **kwargs):
#         """
#         """
#         mean, covar = randvar.mean(), randvar.cov()
#         if np.isscalar(mean) and np.isscalar(covar):
#             mean, covar = mean*np.ones(1), covar*np.eye(1)
#         diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
#         jacob = self.dynamod.jacobian(start, mean, **kwargs)
#         mpred = self.dynamod.dynamics(start, mean, **kwargs)
#         crosscov = covar @ jacob.T
#         cpred = jacob @ crosscov + diffmat
#         return RandomVariable(distribution=Normal(mpred, cpred)), crosscov
#
#     def _predict_continuous(self, start, stop, randvar, **kwargs):
#         """
#         The cont. models that are allowed here all have an
#         implementation of chapman-kolmogorov.
#         For nonlinear SDE models, you would have to implement
#         the ODE for mean and covariance yourself using
#         the jacobian.
#         """
#         step = ((stop - start) / self._nsteps)
#         return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar,
#                                                    **kwargs)
#
#     def update(self, time, randvar, data, **kwargs):
#         """
#         Only discrete measurement models reach this point.
#
#         Hence, the update is straightforward.
#         """
#         return self._update_discrete(time, randvar, data, **kwargs)
#
#     def _update_discrete(self, time, randvar, data, **kwargs):
#         """
#         """
#         mpred, cpred = randvar.mean(), randvar.cov()
#         if np.isscalar(mpred) and np.isscalar(cpred):
#             mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
#         jacob = self.measurementmodel.jacobian(time, mpred,  **kwargs)
#         meascov = self.measurementmodel.diffusionmatrix(time,  **kwargs)
#         meanest = self.measurementmodel.dynamics(time, mpred,  **kwargs)
#         covest = jacob @ cpred @ jacob.T + meascov
#         ccest = cpred @ jacob.T
#         mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
#         cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
#         return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest
#
#
# def _is_discrete(model):
#     """
#     Checks whether the underlying model is discrete.
#     Used in determining whether to use discrete or
#     continuous KF update.
#     """
#     return issubclass(type(model), DiscreteGaussianModel)
#
#
# def _is_not_linear(model):
#     """
#     If model is neither discrete Gaussian or continuous linear,
#     it returns false.
#     """
#     if issubclass(type(model), LinearSDEModel):
#         return False
#     elif issubclass(type(model), DiscreteGaussianModel):
#         return False
#     return True
