"""
Gaussian process interface.
"""

from probnum.prob.randomprocess.randomprocess import _AbstractRandomProcess
from probnum.prob.randomprocess.covariance import *
from probnum.prob import Normal, RandomVariable


class GaussianProcess(_AbstractRandomProcess):
    """
    Parameters
    ----------
    meanfun : callable, signature=``(x)``

    covkernfun : callable or Kernel
        Covariance kernel of the GP. Not to be confused with the
        method :meth:`covfun`.
    """
    def __init__(self, meanfun=None, covkernfun=None, shape=None, dtype=None):
        """
        """
        if not issubclass(type(covkernfun), Covariance):
            covkernfun = Covariance(covfun=covkernfun)
        self._meanfun = meanfun
        self._covfun = covkernfun
        # todo: make bounds [(-inf, inf), ..., (-inf, inf)] depending
        #  on the support/signature of the GP
        super().__init__(shape=shape, dtype=dtype)

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        if self._meanfun is not None and self._covfun is not None:
            dist = Normal(self.meanfun(x), self.covkernfun(x, x))
            return RandomVariable(distribution=dist)
        else:
            raise NotImplementedError("Mean and kernels of the "
                                      "GP are not implemented.")

    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def covkernfun(self, x1, x2):
        """
        Covariance kernel (function) of the Gaussian process.
        """
        raise NotImplementedError


    def covfun(self, x):
        """
        Covariance (function) of the random process.

        Returns covariance of the process at point ``x``.
        """
        return self.covkernfun(x, x)

    def sample(self, size=(), x=None, **kwargs):
        """
        Draw realizations from the random process.
        """
        if x is None:
            raise ValueError("Please specify a location x.")
        else:
            randvar = self.__call__(x)
            return randvar.sample(size=size)