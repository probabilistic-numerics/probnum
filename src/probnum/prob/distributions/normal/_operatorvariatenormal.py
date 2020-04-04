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




class _OperatorvariateNormal(_Normal):
    """
    A normal distribution over finite dimensional linear operators.
    """

    def __init__(self, mean, cov, random_state=None):
        # Check parameters
        self._mean_dim = np.prod(mean.shape)

        # Kronecker structured covariance
        if isinstance(cov, linops.Kronecker):
            m, n = mean.shape
            # If mean has dimension (m x n) then covariance factors must be (m x m) and (n x n)
            if m != cov.A.shape[0] or m != cov.A.shape[1] or n != cov.B.shape[0] or n != cov.B.shape[1]:
                raise ValueError(
                    "Kronecker structured covariance must have factors with the same shape as the mean.")
        # Symmetric Kronecker structured covariance
        elif isinstance(cov, linops.SymmetricKronecker):
            m, n = mean.shape
            # Mean has to be square. If mean has dimension (n x n) then covariance factors must be (n x n).
            if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
                raise ValueError(
                    "Normal distributions with symmetric Kronecker structured covariance must have square mean"
                    + " and square covariance factors with matching dimensions."
                )
        # General case
        elif self._mean_dim != cov.shape[0] or self._mean_dim != cov.shape[1]:
            raise ValueError(
                "Shape mismatch of mean and covariance."
            )

        # Superclass initiator
        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def var(self):
        return linops.Diagonal(Op=self.cov())

    # TODO: implement more efficient versions of (pdf, logpdf, sample) functions for linear operators without todense()
    def _params_todense(self):
        """Returns the mean and covariance of a distribution as dense matrices."""
        if isinstance(self.mean(), linops.LinearOperator):
            mean = self.mean().todense()
        else:
            mean = self.mean()
        if isinstance(self.cov(), linops.LinearOperator):
            cov = self.cov().todense()
        else:
            cov = self.cov()
        return mean, cov

    def pdf(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    def sample(self, size=()):
        mean, cov = self._params_todense()
        samples_ravelled = scipy.stats.multivariate_normal.rvs(mean=mean.ravel(),
                                                               cov=cov,
                                                               size=size,
                                                               random_state=self.random_state)
        return samples_ravelled.reshape(samples_ravelled.shape[:-1] + self.mean().shape)

    def reshape(self, shape):
        raise NotImplementedError

    # Arithmetic Operations
    # TODO: implement special rules for matrix-variate RVs and Kronecker structured covariances
    #  (see e.g. p.64 Thm. 2.3.10 of Gupta: Matrix-variate Distributions)
    def __matmul__(self, other):
        if isinstance(other, Dirac):
            othermean = other.mean()
            delta = linops.Kronecker(linops.Identity(othermean.shape[0]), othermean)
            return Normal(mean=self.mean() @ othermean,
                          cov=delta.T @ (self.cov() @ delta),
                          random_state=self.random_state)
        return NotImplemented


class _SymmetricKroneckerIdenticalFactorsNormal(_OperatorvariateNormal):
    """
    Normal distribution with symmetric Kronecker structured
    covariance with identical factors V (x)_s V.
    """

    def __init__(self, mean, cov, random_state=None):
        m, self._n = mean.shape
        # Mean has to be square. If mean has dimension (n x n) then covariance factors must be (n x n).
        if m != self._n or self._n != cov.A.shape[0] or self._n != cov.B.shape[1]:
            raise ValueError("Normal distributions with symmetric Kronecker"
                             "structured covariance must have square mean and"
                             "square covariance factors with matching"
                             "dimensions.")

        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def sample(self, size=()):
        # Draw standard normal samples
        if np.isscalar(size):
            size = [size]
        size_sample = [self._n * self._n] + list(size)
        stdnormal_samples = scipy.stats.norm.rvs(size=size_sample,
                                                 random_state=self.random_state)

        # Cholesky decomposition
        eps = 10 ** - 12  # TODO: damping needed to avoid negative definite covariances
        cholA = scipy.linalg.cholesky(self.cov().A.todense() + eps * np.eye(self._n), lower=True)

        # Scale and shift
        # TODO: can we avoid todense here and just return operator samples?
        if isinstance(self.mean(), scipy.sparse.linalg.LinearOperator):
            mean = self.mean().todense()
        else:
            mean = self.mean()

        # Appendix E: Bartels, S., Probabilistic Linear Algebra, PhD Thesis 2019
        samples_scaled = (linops.Symmetrize(dim=self._n) @ (
                linops.Kronecker(A=cholA, B=cholA) @ stdnormal_samples))

        return mean[None, :, :] + samples_scaled.T.reshape(-1, self._n, self._n)
