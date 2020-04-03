"""
Normal distribution.

This module implements the Gaussian distribution in its univariate, multi-variate, matrix-variate and operator-variate
forms.
"""
import operator

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.linalg import linops
from probnum.prob.distributions.distribution import Distribution
from probnum.prob.distributions.dirac import Dirac


class Normal(Distribution):
    """
    The (multi-variate) normal distribution.

    The Gaussian distribution is ubiquitous in probability theory, since it is the final and stable or equilibrium
    distribution to which other distributions gravitate under a wide variety of smooth operations, e.g.,
    convolutions and stochastic transformations. One example of this is the central limit theorem. The Gaussian
    distribution is also attractive from a numerical point of view as it is maintained through many transformations
    (e.g. it is stable).

    Parameters
    ----------
    mean : float or array-like or LinearOperator
        Mean of the normal distribution.

    cov : float or array-like or LinearOperator
        (Co-)variance of the normal distribution.

    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to use for drawing
        realizations from this distribution.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    Distribution : Class representing general probability distributions.

    Examples
    --------
    >>> from probnum.prob import Normal
    >>> N = Normal(mean=0.5, cov=1.0)
    >>> N.parameters
    {'mean': 0.5, 'cov': 1.0}
    """

    # TODO: only keep Cholesky factors as covariance to avoid losing symmetry

    def __new__(cls, mean=0., cov=1., random_state=None):
        # Factory method for Normal subclasses
        if cls is Normal:
            # Check input for univariate, multivariate, matrix-variate or operator-variate
            dim1shapes = [(1, 1), (1,), ()]
            if (np.isscalar(mean) and np.isscalar(cov)) or (
                    np.shape(mean) in dim1shapes and np.shape(cov) in dim1shapes):
                return super(Normal, cls).__new__(_UnivariateNormal)
            elif isinstance(mean, (np.ndarray, scipy.sparse.spmatrix,)) and isinstance(cov,
                                                                                       (np.ndarray,
                                                                                        scipy.sparse.spmatrix)):
                if len(mean.shape) == 1:
                    return super(Normal, cls).__new__(_MultivariateNormal)
                else:
                    return super(Normal, cls).__new__(_MatrixvariateNormal)
            elif isinstance(mean, scipy.sparse.linalg.LinearOperator) or isinstance(cov,
                                                                                    scipy.sparse.linalg.LinearOperator):
                if isinstance(cov, linops.SymmetricKronecker) and cov._ABequal:
                    return super(Normal, cls).__new__(_SymmetricKroneckerIdenticalFactorsNormal)
                else:
                    return super(Normal, cls).__new__(_OperatorvariateNormal)
            else:
                raise ValueError(
                    "Cannot instantiate normal distribution with mean of type {} and covariance of type {}.".format(
                        mean.__class__.__name__, cov.__class__.__name__))
        else:
            return super(Normal, cls).__new__(cls, mean=mean, cov=cov, random_state=random_state)

    def __init__(self, mean=0., cov=1., random_state=None):
        # Set dtype to float
        _dtype = float

        # Call to super class initiator
        super().__init__(parameters={"mean": mean, "cov": cov}, dtype=_dtype, random_state=random_state)

    def median(self):
        return self.parameters["mean"]

    def mode(self):
        return self.parameters["mean"]

    def mean(self):
        return self.parameters["mean"]

    def cov(self):
        return self.parameters["cov"]

    def var(self):
        raise NotImplementedError

    # def reshape(self, shape):
    #     try:
    #         # Reshape mean and covariance
    #         self._parameters["mean"].reshape(shape=shape)
    #         # self._parameters["cov"].
    #     except ValueError:
    #         raise ValueError("Cannot reshape this Normal distribution to the given shape: {}".format(str(shape)))

    # Binary arithmetic operations
    def __add__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            return Normal(mean=self.mean() + delta,
                          cov=self.cov(),
                          random_state=self.random_state)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Dirac):
            return self + (-other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            if delta == 0:
                return Dirac(support=0 * self.mean(), random_state=self.random_state)
            else:
                return Normal(mean=self.mean() * delta,
                              cov=self.cov() * delta ** 2,
                              random_state=self.random_state)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division by zero not supported.")
        else:
            if isinstance(other, Dirac):
                return self * operator.inv(other)
            else:
                return NotImplemented

    def __pow__(self, power, modulo=None):
        return NotImplemented

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        if isinstance(other, Dirac):
            return self + other
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Dirac):
            return operator.neg(self) + other
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Dirac):
            return self * other
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            return Normal(mean=delta @ self.mean(),
                          cov=delta @ (self.cov() @ delta.transpose()),
                          random_state=self.random_state)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Dirac):
            return operator.inv(self) * other
        else:
            return NotImplemented

    def __rpow__(self, power, modulo=None):
        return NotImplemented

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    def __iadd__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __imatmul__(self, other):
        return NotImplemented

    def __itruediv__(self, other):
        return NotImplemented

    def __ipow__(self, power, modulo=None):
        return NotImplemented

    # Unary arithmetic operations
    def __neg__(self):
        try:
            return Normal(mean=- self.mean(),
                          cov=self.cov(),
                          random_state=self.random_state)
        except Exception:
            return NotImplemented

    def __pos__(self):
        try:
            return Normal(mean=operator.pos(self.mean()),
                          cov=self.cov(),
                          random_state=self.random_state)
        except Exception:
            return NotImplemented

    def __abs__(self):
        try:
            # todo: add absolute moments of normal (see: https://arxiv.org/pdf/1209.4340.pdf)
            return Distribution(parameters={},
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            return NotImplemented

    def __invert__(self):
        try:
            return Distribution(parameters={},
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            return NotImplemented


class _UnivariateNormal(Normal):
    """The univariate normal distribution."""

    def __init__(self, mean=0., cov=1., random_state=None):
        super().__init__(mean=float(mean), cov=float(cov), random_state=random_state)

    def var(self):
        return self.cov()

    def pdf(self, x):
        return scipy.stats.norm.pdf(x, loc=self.mean(), scale=self.std())

    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mean(), scale=self.std())

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, loc=self.mean(), scale=self.std())

    def logcdf(self, x):
        return scipy.stats.norm.logcdf(x, loc=self.mean(), scale=self.std())

    def sample(self, size=()):
        return scipy.stats.norm.rvs(loc=self.mean(), scale=self.std(), size=size, random_state=self.random_state)

    def reshape(self, shape):
        raise NotImplementedError

    # Arithmetic Operations
    def __matmul__(self, other):
        if isinstance(other, Dirac):
            delta = other.mean()
            return Normal(mean=np.squeeze(self.mean() @ delta),
                          cov=np.squeeze(delta @ (self.cov() @ delta.transpose())),
                          random_state=self.random_state)
        # TODO: implement special rules for matrix-variate RVs and Kronecker structured covariances
        #  (see e.g. p.64 Thm. 2.3.10 of Gupta: Matrix-variate distribution)
        return NotImplemented


class _MultivariateNormal(Normal):
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


class _MatrixvariateNormal(Normal):
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


class _OperatorvariateNormal(Normal):
    """A normal distribution over finite dimensional linear operators."""

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
    """Normal distribution with symmetric Kronecker structured covariance with identical factors V (x)_s V."""

    def __init__(self, mean, cov, random_state=None):
        m, self._n = mean.shape
        # Mean has to be square. If mean has dimension (n x n) then covariance factors must be (n x n).
        if m != self._n or self._n != cov.A.shape[0] or self._n != cov.B.shape[1]:
            raise ValueError(
                "Normal distributions with symmetric Kronecker structured covariance must have square mean"
                + " and square covariance factors with matching dimensions."
            )

        super().__init__(mean=mean, cov=cov, random_state=random_state)

    def sample(self, size=()):
        # Draw standard normal samples
        if np.isscalar(size):
            size = [size]
        size_sample = [self._n * self._n] + list(size)
        stdnormal_samples = scipy.stats.norm.rvs(size=size_sample, random_state=self.random_state)

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
