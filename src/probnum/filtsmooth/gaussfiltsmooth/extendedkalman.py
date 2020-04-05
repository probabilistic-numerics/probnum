"""
"""
import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import gaussianfilter
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace.continuous import LinearSDEModel
from probnum.filtsmooth.statespace.discrete import DiscreteGaussianModel

__all__ = ["ExtendedKalmanFilter"]


class ExtendedKalmanFilter(gaussianfilter.GaussianFilter):
    """
    """

    def __init__(self, dynamod, measmod, initdist, _nsteps=15):
        """
        dynmod: continuousmodel.linear.Linear or subclass
        measmod: discretemodel.gaussmarkov.Nonlinear or subclass
        initdist : interpolating.MultivariateGaussian
        _nsteps : approximate integration.

        Functionality so far restricted to linear SDEs because these implement
        solutions to the CKE. Support of nonlinear equations can be supported
        through overwriting the "update" step below and hard-coding the solution
        of the differential equation in Linear.chapmankolmogorov().
        """
        if _is_not_linear(dynamod):
            raise TypeError(
                "Extended Kalman filter doesn't support dynamic model")
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise TypeError(
                "Extended Kalman filter doesn't support measurement model")
        if not issubclass(type(initdist.distribution), Normal):
            raise TypeError("Extended Kalman filter doesn't support "
                            "initial distribution")
        self.dynamod = dynamod
        self.measmod = measmod
        self.initdist = initdist
        self._nsteps = _nsteps

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
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean*np.ones(1), covar*np.eye(1)
        diffmat = self.dynamod.diffusionmatrix(start, *args, **kwargs)
        jacob = self.dynamod.jacobian(start, mean, *args, **kwargs)
        mpred = self.dynamod.dynamics(start, mean, *args, **kwargs)
        cpred = jacob @ covar @ jacob.T + diffmat
        return RandomVariable(distribution=Normal(mpred, cpred))

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
        mpred, cpred = randvar.mean(), randvar.cov()
        if np.isscalar(mpred) and np.isscalar(cpred):
            mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
        jacob = self.measurementmodel.jacobian(time, mpred, *args, **kwargs)
        meascov = self.measurementmodel.diffusionmatrix(time, *args, **kwargs)
        meanest = self.measurementmodel.dynamics(time, mpred, *args, **kwargs)
        covest = jacob @ cpred @ jacob.T + meascov
        ccest = cpred @ jacob.T
        mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
        cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
        return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest


def _is_discrete(model):
    """
    Checks whether the underlying model is discrete.
    Used in determining whether to use discrete or
    continuous KF update.
    """
    return issubclass(type(model), DiscreteGaussianModel)


def _is_not_linear(model):
    """
    If model is neither discrete Gaussian or continuous linear,
    it returns false.
    """
    if issubclass(type(model), LinearSDEModel):
        return False
    elif issubclass(type(model), DiscreteGaussianModel):
        return False
    return True
