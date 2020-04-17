import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import gaussfiltsmooth, unscentedtransform
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace.continuous import LinearSDEModel
from probnum.filtsmooth.statespace.discrete import DiscreteGaussianModel, DiscreteGaussianLinearModel



class UnscentedKalmanSmoother(gaussfiltsmooth.GaussianSmoother):
    """
    Kalman smoother as a simple add-on to the filtering.
    The rest is implemented in GaussianSmoother.
    """

    def __init__(self, dynamod, measmod, initdist, alpha, beta, kappa,
                 _nsteps=15):
        """ """
        unscentedkalfilt = UnscentedKalmanFilter(dynamod, measmod, initdist, alpha, beta, kappa,
                 _nsteps)
        super().__init__(unscentedkalfilt)



class UnscentedKalmanFilter(gaussfiltsmooth.GaussianFilter):
    """
    """

    def __init__(self, dynamod, measmod, initdist, alpha, beta, kappa,
                 _nsteps=15):
        """
        dynmod: continuousmodel.linear.Linear or subclass
        measmod: discretemodel.gaussmarkov.Nonlinear or subclass
        initdist : polynomial.MultivariateGaussian
        _nsteps : approximate integration.

        Functionality so far restricted to linear SDEs because these implement
        solutions to the CKE. Support of nonlinear equations can be supported
        through overwriting the "update" step below and hard-coding the solution
        of the differential equation in Linear.chapmankolmogorov().
        """
        if _is_not_gaussian(dynamod):
            raise TypeError("Unscented Kalman filter doesn't "
                            "support dynamic model")
        if not issubclass(type(measmod), DiscreteGaussianModel):
            raise TypeError("Extended Kalman filter doesn't "
                            "support measurement model")
        if not issubclass(type(initdist.distribution), Normal):
            raise TypeError("Extended Kalman filter doesn't "
                            "support initial distribution")
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
        if issubclass(type(self.dynamod), DiscreteGaussianLinearModel):
            return self._predict_discrete_linear(start, randvar, *args,
                                                 **kwargs)
        else:
            return self._predict_discrete_nonlinear(start, randvar, *args,
                                                    **kwargs)

    def _predict_discrete_linear(self, start, randvar, *args, **kwargs):
        """
        """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean*np.ones(1), covar*np.eye(1)
        dynamat = self.dynamod.dynamicsmatrix(start, *args, **kwargs)
        forcevec = self.dynamod.force(start, *args, **kwargs)
        diffmat = self.dynamod.diffusionmatrix(start, *args, **kwargs)
        mpred = dynamat @ mean + forcevec
        crosscov = covar @ dynamat.T
        cpred = dynamat @ crosscov + diffmat
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

    def _predict_discrete_nonlinear(self, start, randvar, *args, **kwargs):
        """
        Executes unscented transform!
        """
        mean, covar = randvar.mean(), randvar.cov()
        if np.isscalar(mean) and np.isscalar(covar):
            mean, covar = mean*np.ones(1), covar*np.eye(1)
        sigmapts = self.ut.sigma_points(mean, covar)
        proppts = self.ut.propagate(start, sigmapts, self.dynamod.dynamics)
        diffmat = self.dynamod.diffusionmatrix(start, *args, **kwargs)
        mpred, cpred, crosscov = self.ut.estimate_statistics(proppts, sigmapts,
                                                       diffmat, mean)
        return RandomVariable(distribution=Normal(mpred, cpred)), crosscov

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
        if issubclass(type(self.dynamod), DiscreteGaussianLinearModel):
            return self._update_discrete_linear(time, randvar, data, *args,
                                                **kwargs)
        else:
            return self._update_discrete_nonlinear(time, randvar, data, *args,
                                                   **kwargs)

    def _update_discrete_linear(self, time, randvar, data, *args, **kwargs):
        """
        """
        mpred, cpred = randvar.mean(), randvar.cov()
        if np.isscalar(mpred) and np.isscalar(cpred):
            mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
        measmat = self.measurementmodel.dynamicsmatrix(time, *args, **kwargs)
        meascov = self.measurementmodel.diffusionmatrix(time, *args, **kwargs)
        meanest = measmat @ mpred
        covest = measmat @ cpred @ measmat.T + meascov
        ccest = cpred @ measmat.T
        mean = mpred + ccest @ np.linalg.solve(covest, data - meanest)
        cov = cpred - ccest @ np.linalg.solve(covest.T, ccest.T)
        return RandomVariable(distribution=Normal(mean, cov)), covest, ccest, meanest

    def _update_discrete_nonlinear(self, time, randvar, data, *args, **kwargs):
        """
        """
        mpred, cpred = randvar.mean(), randvar.cov()
        if np.isscalar(mpred) and np.isscalar(cpred):
            mpred, cpred = mpred*np.ones(1), cpred*np.eye(1)
        sigmapts = self.ut.sigma_points(mpred, cpred)
        proppts = self.ut.propagate(time, sigmapts, self.measmod.dynamics)
        meascov = self.measmod.diffusionmatrix(time, *args, **kwargs)
        meanest, covest, ccest = self.ut.estimate_statistics(proppts, sigmapts,
                                                        meascov, mpred)
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


def _is_not_gaussian(model):
    """
    If model is neither discrete Gaussian or continuous linear,
    it returns false.
    """
    if issubclass(type(model), LinearSDEModel):
        return False
    elif issubclass(type(model), DiscreteGaussianModel):
        return False
    return True

