"""
Multivariate normal distribution class.

It is internal. For public use, refer to normal.Normal instead.
"""

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.prob.distributions.dirac import Dirac
from probnum.prob.distributions.normal._normal import _Normal
from probnum.prob.distributions.normal._univariatenormal import _UnivariateNormal

class _MultivariateNormal(_Normal):
    """
    The multivariate normal distribution.
    """

    def __init__(self, mean, cov, random_state=None):
        """
        Checks if mean and covariance have matching shapes before
        initialising.
        """
        meandim = np.prod(mean.shape)
        if len(cov.shape) != 2:
            raise ValueError("Covariance must be a 2D matrix"
                             "or linear operator.")
        if meandim != cov.shape[0] or meandim != cov.shape[1]:
            raise ValueError("Shape mismatch of mean and covariance. Total "
                             "number of elements of the mean must match the "
                             "first and second dimension of the covariance.")
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return np.diag(self.cov())

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean(),
                                                   cov=self.cov())

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean(),
                                                      cov=self.cov())

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.mean(),
                                                   cov=self.cov())

    def logcdf(self, x):
        return scipy.stats.multivariate_normal.logcdf(x, mean=self.mean(),
                                                      cov=self.cov())

    def sample(self, size=()):
        return scipy.stats.multivariate_normal.rvs(mean=self.mean(),
                                                   cov=self.cov(), size=size,
                                                   random_state=self.random_state)

    def reshape(self, shape):
        raise NotImplementedError

    # Arithmetic Operations ###############################

    def __matmul__(self, other):
        """
        Only works if other is a Dirac, which implies
        matrix-multiplication with some constant.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            newmean = self.mean() @ delta
            newcov = delta.T @ (self.cov() @ delta)
            if np.isscalar(newmean) and np.isscalar(newcov):
                return _UnivariateNormal(mean=newmean, cov=newcov,
                                         random_state=self.random_state)
            else:
                return _MultivariateNormal(mean=newmean, cov=newcov,
                                           random_state=self.random_state)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        Only works if other is a Dirac, which implies
        matrix-multiplication with some constant.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            newmean = delta @ self.mean()
            newcov = delta @ (self.cov() @ delta.T)
            if np.isscalar(newmean) and np.isscalar(newcov):
                return _UnivariateNormal(mean=newmean, cov=newcov,
                                         random_state=self.random_state)
            else:
                return _MultivariateNormal(mean=newmean, cov=newcov,
                                           random_state=self.random_state)
        else:
            return NotImplemented
