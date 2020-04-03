"""
Univariate normal class.
"""
import operator
import abc

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.linalg import linops
from probnum.prob.distributions.distribution import Distribution
from probnum.prob.distributions.dirac import Dirac

# Import "private" convenience modules
from probnum.prob.distributions.normal._normal import _Normal




class _MultivariateNormal(_Normal):
    """The multivariate normal distribution."""

    def __init__(self, mean, cov, random_state=None):

        # Check parameters
        _mean_dim = np.prod(mean.shape)
        if len(cov.shape) != 2:
            raise ValueError("Covariance must be a 2D matrix or linear operator.")
        if _mean_dim != cov.shape[0] or _mean_dim != cov.shape[1]:
            raise ValueError(
                "Shape mismatch of mean and covariance. Total number of elements of the mean must match " +
                "the first and second dimension of the covariance.")

        # Superclass initiator
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return np.diag(self.cov())

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean(), cov=self.cov())

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean(), cov=self.cov())

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.mean(), cov=self.cov())

    def logcdf(self, x):
        return scipy.stats.multivariate_normal.logcdf(x, mean=self.mean(), cov=self.cov())

    def sample(self, size=()):
        return scipy.stats.multivariate_normal.rvs(mean=self.mean(), cov=self.cov(), size=size,
                                                   random_state=self.random_state)

    def reshape(self, shape):
        raise NotImplementedError

    # Arithmetic Operations
    def __matmul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            return Normal(mean=self.mean() @ delta,
                          cov=delta.T @ (self.cov() @ delta),
                          random_state=self.random_state)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            return Normal(mean=delta @ self.mean(),
                          cov=delta @ (self.cov() @ delta.T),
                          random_state=self.random_state)
        return NotImplemented


class _MatrixvariateNormal(_Normal):
    """The matrixvariate normal distribution."""

    def __init__(self, mean, cov, random_state=None):

        # Check parameters
        _mean_dim = np.prod(mean.shape)
        if len(cov.shape) != 2:
            raise ValueError("Covariance must be a 2D matrix.")
        if _mean_dim != cov.shape[0] or _mean_dim != cov.shape[1]:
            raise ValueError(
                "Shape mismatch of mean and covariance. Total number of elements of the mean must match " +
                "the first and second dimension of the covariance.")

        # Superclass initiator
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return np.diag(self.cov())

    def pdf(self, x):
        # TODO: need to reshape x into number of matrices given
        pdf_ravelled = scipy.stats.multivariate_normal.pdf(x.ravel(),
                                                           mean=self.mean().ravel(),
                                                           cov=self.cov())
        # TODO: this reshape is incorrect, write test for multiple matrices
        return pdf_ravelled.reshape(shape=self.mean().shape)

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    def sample(self, size=()):
        samples_ravelled = scipy.stats.multivariate_normal.rvs(mean=self.mean().ravel(),
                                                               cov=self.cov(),
                                                               size=size,
                                                               random_state=self.random_state)
        # TODO: maybe distributions need an attribute sample_shape
        return samples_ravelled.reshape(shape=self.mean().shape)

    def reshape(self, shape):
        raise NotImplementedError

    # Arithmetic Operations
    # TODO: implement special rules for matrix-variate RVs and Kronecker structured covariances
    #  (see e.g. p.64 Thm. 2.3.10 of Gupta: Matrix-variate Distributions)

    def __matmul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            raise NotImplementedError
        # TODO: implement generic:
        return NotImplemented






