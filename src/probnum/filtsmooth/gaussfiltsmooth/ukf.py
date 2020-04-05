import numpy as np

from diffeq.bayesianfilter.gaussianfilter import gaussianfilter, \
    unscentedtransform
from diffeq.randomvariable import gaussian
from diffeq.statespace.continuousmodel import linearcontinuous
from diffeq.statespace.discretemodel import gaussmarkov



np.random.seed(1)

class UnscentedKalmanFilter(gaussianfilter.GaussianFilter):
    """
    """

    def __init__(self, dynamod, measmod, initdist, alpha, beta, kappa,
                 _nsteps=5):
        """
        dynmod: continuousmodel.linear.Linear or subclass
        measmod: discretemodel.gaussmarkov.Nonlinear or subclass
        initdist : gaussian.MultivariateGaussian
        _nsteps : approximate integration.

        Functionality so far restricted to linear SDEs because these implement
        solutions to the CKE. Support of nonlinear equations can be supported
        through overwriting the "update" step below and hard-coding the solution
        of the differential equation in Linear.chapmankolmogorov().
        """
        if _is_not_gaussian(dynamod):
            raise TypeError(
                "Extended Kalman filter doesn't support dynamic model")
        if not issubclass(type(measmod), gaussmarkov.Nonlinear):
            raise TypeError(
                "Extended Kalman filter doesn't support measurement model")
        if not issubclass(type(initdist), gaussian.MultivariateGaussian):
            raise TypeError(
                "Extended Kalman filter doesn't support initial distribution")
        self.dynamod = dynamod
        self.measmod = measmod
        self.initdist = initdist
        self._nsteps = _nsteps
        self.ut = unscentedtransform.UnscentedTransform(self.dynamod.ndim,
                                                        alpha, beta, kappa)

    @property
    def dynamicmodel(self):
        """ """
        return self.dynamod

    @property
    def measurementmodel(self):
        """ """
        return self.measmod

    @property
    def initialdistribution(self):
        """ """
        return self.initdist

    def predict(self, start, stop, randvar, *args, **kwargs):
        """
        """
        if _is_discrete(self.dynamod):
            return self._predict_discrete(start, randvar, *args, **kwargs)
        else:
            return self._predict_continuous(start, stop, randvar, *args,
                                            **kwargs)

    def _predict_discrete(self, start, randvar, *args, **kwargs):
        """
        """
        if issubclass(type(self.dynamod), gaussmarkov.Linear):
            return self._predict_discrete_linear(start, randvar, *args,
                                                 **kwargs)
        else:
            return self._predict_discrete_nonlinear(start, randvar, *args,
                                                    **kwargs)

    def _predict_discrete_linear(self, start, randvar, *args, **kwargs):
        """
        """
        mean, covar = randvar.mean, randvar.covar
        dynamat = self.dynamod.dynamicsmatrix(start, *args, **kwargs)
        forcevec = self.dynamod.force(start, *args, **kwargs)
        diffmat = self.dynamod.diffusionmatrix(start, *args, **kwargs)
        mpred = dynamat @ mean + forcevec
        cpred = dynamat @ covar @ dynamat.T + diffmat
        return gaussian.MultivariateGaussian(mpred, cpred)

    def _predict_discrete_nonlinear(self, start, randvar, *args, **kwargs):
        """
        Executes unscented transform!
        """
        mean, covar = randvar.mean, randvar.covar
        sigmapts = self.ut.sigma_points(mean, covar)
        proppts = self.ut.propagate(start, sigmapts, self.dynamod.dynamics)
        diffmat = self.dynamod.diffusionmatrix(start, *args, **kwargs)
        mpred, cpred, __ = self.ut.estimate_statistics(proppts, sigmapts,
                                                       diffmat, mean)
        return gaussian.MultivariateGaussian(mpred, cpred)

    def _predict_continuous(self, start, stop, randvar, *args, **kwargs):
        """
        The cont. models that are allowed here all have an
        implementation of chapman-kolmogorov.
        For nonlinear SDE models, you would have to implement
        the ODE for mean and covariance yourself using
        the jacobian.
        """
        step = ((stop - start) / self._nsteps)
        return self.dynamicmodel.chapmankolmogorov(start, stop, step, randvar,
                                                   *args, **kwargs)

    def update(self, time, randvar, data, *args, **kwargs):
        """
        Only discrete measurement models reach this point.

        Hence, the update is straightforward.
        """
        return self._update_discrete(time, randvar, data, *args, **kwargs)

    def _update_discrete(self, time, randvar, data, *args, **kwargs):
        """
        """
        if issubclass(type(self.dynamod), gaussmarkov.Linear):
            return self._update_discrete_linear(time, randvar, data, *args,
                                                **kwargs)
        else:
            return self._update_discrete_nonlinear(time, randvar, data, *args,
                                                   **kwargs)

    def _update_discrete_linear(self, time, randvar, data, *args, **kwargs):
        """
        """
        mpred, cpred = randvar.mean, randvar.covar
        measmat = self.measurementmodel.dynamicsmatrix(time, *args, **kwargs)
        meascov = self.measurementmodel.diffusionmatrix(time, *args, **kwargs)
        meanest = measmat @ mpred
        covest = measmat @ cpred @ measmat.T + meascov
        ccest = cpred @ measmat.T
        mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
        cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
        return gaussian.MultivariateGaussian(mean, cov), covest, ccest, meanest

    def _update_discrete_nonlinear(self, time, randvar, data, *args, **kwargs):
        """
        """
        mpred, cpred = randvar.mean, randvar.covar
        sigmapts = self.ut.sigma_points(mpred, cpred)
        proppts = self.ut.propagate(time, sigmapts, self.measmod.dynamics)
        meascov = self.measmod.diffusionmatrix(time, *args, **kwargs)
        meanest, covest, ccest = self.ut.estimate_statistics(proppts, sigmapts,
                                                        meascov, mpred)
        mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
        cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
        return gaussian.MultivariateGaussian(mean, cov), covest, ccest, meanest


def _is_discrete(model):
    """
    Checks whether the underlying model is discrete.
    Used in determining whether to use discrete or
    continuous KF update.
    """
    return issubclass(type(model), gaussmarkov.Nonlinear)


def _is_not_gaussian(model):
    """
    If model is neither discrete Gaussian or continuous linear,
    it returns false.
    """
    if issubclass(type(model), linearcontinuous.Linear):
        return False
    elif issubclass(type(model), gaussmarkov.Nonlinear):
        return False
    return True

#
#
#
#
#
#
# """
# This implementation has not been run even once
# and is more of a draft. Do not expect anything
# to be correct here.
# """
#
# import numpy as np
#
# from diffeq.auxiliary.randomvariable import gaussian
# from diffeq.bayesianfilter import gaussfiltsmooth
# from diffeq.bayesianfilter.gaussfiltsmooth import unscentedtransform
# from diffeq.archive.archive import additivegaussian
#
#
# class UnscentedKalmanFilter(gaussfiltsmooth.GaussianFilter):
#     """
#     """
#
#     def __init__(self, nonlineargaussian, alpha, beta, kappa):
#         """
#         Input is a NonLinearGaussian statespace model.
#         """
#         if not issubclass(type(nonlineargaussian.dynamicmodel), additivegaussian.NonLinearGaussian):
#             raise TypeError("Extended Kalman filter needs a linearised state space model.")
#         if not issubclass(type(nonlineargaussian.measurementmodel), additivegaussian.NonLinearGaussian):
#             raise TypeError("Extended Kalman filter needs a linearised state space model.")
#
#         self.nonlineargaussian = nonlineargaussian
#         ndim = self.nonlineargaussian.dynamicmodel.ndim
#         self.ut = unscentedtransform.UnscentedTransform(ndim, alpha, beta, kappa)
#
#     def predict(self, time, randvar, *args, **kwargs):
#         """
#         Input and output are multivariate Gaussian random variables.
#         """
#         mean, covar = randvar.mean, randvar.covar
#         sigmapts = self.ut.sigma_points(mean, covar)
#         proppts = self.ut.propagate(time, sigmapts, self.nonlineargaussian.dynamicmodel.evaluate)
#         meascovmat = self.nonlineargaussian.dynamicmodel.covariance(time, *args, **kwargs)
#         mpred, cpred, __ = self.ut.estimate_statistics(proppts, sigmapts, meascovmat, mean)
#         return gaussian.MultivariateGaussian(mpred, cpred)
#
#     def update(self, time, randvar, data, *args, **kwargs):
#         """
#         Input and output are multivariate Gaussian random variables.
#         data is a 1d array
#
#         Arguments
#         ---------
#         time: float
#             current time
#         randvar: auxiliary.randomvariable.RandomVariable object; d-valued
#             usually a Gaussian
#         data: np.ndarray, shape (d,)
#             current data input
#         """
#         mpred, cpred = randvar.mean, randvar.covar
#         sigmapts = self.ut.sigma_points(mpred, cpred)
#         proppts = self.ut.propagate(time, sigmapts, self.nonlineargaussian.measurementmodel.evaluate)
#         meascovmat = self.nonlineargaussian.measurementmodel.covariance(time, *args, **kwargs)
#         mest, cest, ccest = self.ut.estimate_statistics(proppts, sigmapts, meascovmat, mpred)
#         kalgain = ccest @ np.linalg.inv(cest)  # should be stable, cest is small
#         mean = mpred + kalgain @ (data - mest)
#         covar = cpred - kalgain @ cest @ kalgain.T
#         return gaussian.MultivariateGaussian(mean, covar)
#
#     @property
#     def statespacemodel(self):
#         """
#         """
#         return self.nonlineargaussian
