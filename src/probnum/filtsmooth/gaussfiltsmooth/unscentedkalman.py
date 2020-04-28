import numpy as np

from probnum.filtsmooth.gaussfiltsmooth.gaussfiltsmooth import *
from probnum.prob import RandomVariable, Normal
from probnum.filtsmooth.statespace import *
from probnum.filtsmooth.gaussfiltsmooth.unscentedtransform import *


class UnscentedKalmanSmoother(GaussianSmoother):
    """
    ExtendedKalman smoother as a simple add-on to the filtering.
    The rest is implemented in GaussianSmoother.
    """

    def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
        """ """
        unskalfilt = UnscentedKalmanFilter(dynamod, measmod, initrv,
                                           alpha, beta, kappa, **kwargs)
        super().__init__(unskalfilt)


class UnscentedKalmanFilter:
    """
    Factory method for Unscented Kalman filters.
    """
    def __new__(cls, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
        """ """
        if cls is UnscentedKalmanFilter:
            if _cont_cont(dynamod, measmod):
                return ContContUnscentedKalmanFilter(dynamod, measmod,
                                                    initrv, alpha, beta, kappa, **kwargs)
            if _cont_disc(dynamod, measmod):
                return ContDiscUnscentedKalmanFilter(dynamod, measmod,
                                                    initrv, alpha, beta, kappa, **kwargs)
            if _disc_disc(dynamod, measmod):
                return DiscDiscUnscentedKalmanFilter(dynamod, measmod,
                                                    initrv, alpha, beta, kappa,  **kwargs)
            else:
                errmsg = ("Cannot instantiate Unscented Kalman filter with "
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


class ContContUnscentedKalmanFilter(ContContGaussianFilter,
                                    UnscentedKalmanFilter):
    """
    Not implemented.

    Error is raised in super().__init__().

    If you'd like to implement it, do your magic here.
    """
    def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
        """ """
        raise NotImplementedError("Continuous-Continuous Unscented "
                                  "Kalman Filtering is not implemented")


class ContDiscUnscentedKalmanFilter(ContDiscGaussianFilter,
                                    UnscentedKalmanFilter):
    """
    Completes implementation of ContinuousContinuousGaussianFilter.

    Provides predict() and update() methods.
    """
    def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
        """
        """
        if not issubclass(type(dynamod), LinearSDEModel):
            raise ValueError("This implementation of "
                            "ContDiscUnscentedKalmanFilter "
                            "requires a linear dynamic model.")
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise ValueError("DiscDiscUnscentedKalmanFilter requires "
                            "a Gaussian measurement model.")
        if "cke_nsteps" in kwargs.keys():
            self.cke_nsteps = kwargs["cke_nsteps"]
        else:
            self.cke_nsteps = 1
        super().__init__(dynamod, measmod, initrv)
        self.ut = UnscentedTransform(self.dynamod.ndim, alpha, beta, kappa)

    def predict(self, start, stop, randvar, **kwargs):
        """ """
        step = ((stop - start) / self.cke_nsteps)
        return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar,
                                                   **kwargs)

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_unskalman_update(time, randvar, data, self.measurementmodel, self.ut, **kwargs)


class DiscDiscUnscentedKalmanFilter(DiscDiscGaussianFilter, UnscentedKalmanFilter):
    """
    """
    def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa, **kwargs):
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
        self.ut = UnscentedTransform(self.dynamod.ndim, alpha, beta, kappa)

    def predict(self, start, stop, randvar, **kwargs):
        """
        """
        if issubclass(type(self.dynamod), DiscreteGaussianLinearModel):
            return self._predict_linear(start, randvar, **kwargs)
        else:
            return self._predict_nonlinear(start, randvar, **kwargs)

    def _predict_linear(self, start, randvar, **kwargs):
        """
        Basic Kalman update because model is linear.
        """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean*np.ones(1), covar*np.eye(1)
        dynamat = self.dynamod.dynamicsmatrix(start, **kwargs)
        forcevec = self.dynamod.force(start, **kwargs)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        mpred = dynamat @ mean + forcevec
        crosscov = covar @ dynamat.T
        cpred = dynamat @ crosscov + diffmat
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

    def _predict_nonlinear(self, start, randvar, **kwargs):
        """
        Executes unscented transform!
        """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean*np.ones(1), covar*np.eye(1)
        sigmapts = self.ut.sigma_points(mean, covar)
        proppts = self.ut.propagate(start, sigmapts, self.dynamod.dynamics)
        diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
        mpred, cpred, crosscov = self.ut.estimate_statistics(proppts, sigmapts,
                                                       diffmat, mean)
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

    def update(self, time, randvar, data, **kwargs):
        """ """
        return _discrete_unskalman_update(time, randvar, data, self.measurementmodel, self.ut, **kwargs)


def _discrete_unskalman_update(time, randvar, data, measurementmodel, ut, **kwargs):
    """
    """
    if issubclass(type(measurementmodel), DiscreteGaussianLinearModel):
        return _update_discrete_linear(time, randvar, data, measurementmodel, **kwargs)
    else:
        return _update_discrete_nonlinear(time, randvar, data, measurementmodel, ut, **kwargs)


def _update_discrete_linear(time, randvar, data, measurementmodel, **kwargs):
    """
    """
    mpred, cpred = randvar.mean(), randvar.cov()
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
    measmat = measurementmodel.dynamicsmatrix(time,  **kwargs)
    meascov = measurementmodel.diffusionmatrix(time,  **kwargs)
    meanest = measmat @ mpred
    covest = measmat @ cpred @ measmat.T + meascov
    ccest = cpred @ measmat.T
    mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
    cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
    return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest

def _update_discrete_nonlinear(time, randvar, data, measurementmodel, ut, **kwargs):
    """
    """
    mpred, cpred = randvar.mean(), randvar.cov()
    if np.isscalar(mpred) and np.isscalar(cpred):
        mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
    sigmapts = ut.sigma_points(mpred, cpred)
    proppts = ut.propagate(time, sigmapts, measurementmodel.dynamics)
    meascov = measurementmodel.diffusionmatrix(time,  **kwargs)
    meanest, covest, ccest = ut.estimate_statistics(proppts, sigmapts,
                                                    meascov, mpred)
    mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
    cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
    return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest































#
#
#
#
#
#
#
# class UnscentedKalmanSmoother(gaussfiltsmooth.GaussianSmoother):
#     """
#     Kalman smoother as a simple add-on to the filtering.
#     The rest is implemented in GaussianSmoother.
#     """
#
#     def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa,
#                  _nsteps=15):
#         """ """
#         unscentedkalfilt = UnscentedKalmanFilter(dynamod, measmod, initrv, alpha, beta, kappa,
#                  _nsteps)
#         super().__init__(unscentedkalfilt)
#
#
#
# class UnscentedKalmanFilter(gaussfiltsmooth.GaussianFilter):
#     """
#     """
#
#     def __init__(self, dynamod, measmod, initrv, alpha, beta, kappa,
#                  _nsteps=15):
#         """
#         dynmod: continuousmodel.linear.Linear or subclass
#         measmod: discretemodel.gaussmarkov.Nonlinear or subclass
#         initrv : polynomial.MultivariateGaussian
#         _nsteps : approximate integration.
#
#         Functionality so far restricted to linear SDEs because these implement
#         solutions to the CKE. Support of nonlinear equations can be supported
#         through overwriting the "update" step below and hard-coding the solution
#         of the differential equation in Linear.chapmankolmogorov().
#         """
#         if _is_not_gaussian(dynamod):
#             raise ValueError("Unscented Kalman filter doesn't "
#                             "support dynamic model")
#         if not issubclass(type(measmod), DiscreteGaussianModel):
#             raise ValueError("Extended Kalman filter doesn't "
#                             "support measurement model")
#         if not issubclass(type(initrv.distribution), Normal):
#             raise ValueError("Extended Kalman filter doesn't "
#                             "support initial distribution")
#         self.dynamod = dynamod
#         self.measmod = measmod
#         self.initrv = initrv
#         self._nsteps = _nsteps
#         self.ut = unscentedtransform.UnscentedTransform(self.dynamod.ndim,
#                                                         alpha, beta, kappa)
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
#         if issubclass(type(self.dynamod), DiscreteGaussianLinearModel):
#             return self._predict_discrete_linear(start, randvar,
#                                                  **kwargs)
#         else:
#             return self._predict_discrete_nonlinear(start, randvar,
#                                                     **kwargs)
#
#     def _predict_discrete_linear(self, start, randvar, **kwargs):
#         """
#         """
#         mean, covar = randvar.mean(), randvar.cov()
#         if np.isscalar(mean) and np.isscalar(covar):
#             mean, covar = mean*np.ones(1), covar*np.eye(1)
#         dynamat = self.dynamod.dynamicsmatrix(start, **kwargs)
#         forcevec = self.dynamod.force(start, **kwargs)
#         diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
#         mpred = dynamat @ mean + forcevec
#         crosscov = covar @ dynamat.T
#         cpred = dynamat @ crosscov + diffmat
#         return RandomVariable(distribution=Normal(mpred, cpred)), crosscov
#
#     def _predict_discrete_nonlinear(self, start, randvar, **kwargs):
#         """
#         Executes unscented transform!
#         """
#         mean, covar = randvar.mean(), randvar.cov()
#         if np.isscalar(mean) and np.isscalar(covar):
#             mean, covar = mean*np.ones(1), covar*np.eye(1)
#         sigmapts = self.ut.sigma_points(mean, covar)
#         proppts = self.ut.propagate(start, sigmapts, self.dynamod.dynamics)
#         diffmat = self.dynamod.diffusionmatrix(start, **kwargs)
#         mpred, cpred, crosscov = self.ut.estimate_statistics(proppts, sigmapts,
#                                                        diffmat, mean)
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
#         return self._update_discrete(time, randvar, data,  **kwargs)
#
#     def _update_discrete(self, time, randvar, data,  **kwargs):
#         """
#         """
#         if issubclass(type(self.dynamod), DiscreteGaussianLinearModel):
#             return self._update_discrete_linear(time, randvar, data,
#                                                 **kwargs)
#         else:
#             return self._update_discrete_nonlinear(time, randvar, data,
#                                                    **kwargs)
#
#     def _update_discrete_linear(self, time, randvar, data, **kwargs):
#         """
#         """
#         mpred, cpred = randvar.mean(), randvar.cov()
#         if np.isscalar(mpred) and np.isscalar(cpred):
#             mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
#         measmat = self.measurementmodel.dynamicsmatrix(time,  **kwargs)
#         meascov = self.measurementmodel.diffusionmatrix(time,  **kwargs)
#         meanest = measmat @ mpred
#         covest = measmat @ cpred @ measmat.T + meascov
#         ccest = cpred @ measmat.T
#         mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
#         cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
#         return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest
#
#     def _update_discrete_nonlinear(self, time, randvar, data, **kwargs):
#         """
#         """
#         mpred, cpred = randvar.mean(), randvar.cov()
#         if np.isscalar(mpred) and np.isscalar(cpred):
#             mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
#         sigmapts = self.ut.sigma_points(mpred, cpred)
#         proppts = self.ut.propagate(time, sigmapts, self.measmod.dynamics)
#         meascov = self.measmod.diffusionmatrix(time,  **kwargs)
#         meanest, covest, ccest = self.ut.estimate_statistics(proppts, sigmapts,
#                                                         meascov, mpred)
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
# def _is_not_gaussian(model):
#     """
#     If model is neither discrete Gaussian or continuous linear,
#     it returns false.
#     """
#     if issubclass(type(model), LinearSDEModel):
#         return False
#     elif issubclass(type(model), DiscreteGaussianModel):
#         return False
#     return True
#
