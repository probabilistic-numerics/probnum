from typing import Optional, Union

import numpy as np
import scipy.stats

from probnum import prob, utils as _utils
from probnum.linalg import linops
from probnum.prob import _random_variable


_ValueType = Union[np.floating, np.ndarray, linops.LinearOperator]


class Normal(_random_variable.RandomVariable[np.ndarray]):
    """
    The normal distribution.

    The Gaussian distribution is ubiquitous in probability theory, since
    it is the final and stable or equilibrium distribution to which
    other distributions gravitate under a wide variety of smooth
    operations, e.g., convolutions and stochastic transformations.
    One example of this is the central limit theorem. The Gaussian
    distribution is also attractive from a numerical point of view as it
    is maintained through many transformations (e.g. it is stable).

    Parameters
    ----------
    mean : float or array-like or LinearOperator
        Mean of the normal distribution.

    cov : float or array-like or LinearOperator
        (Co-)variance of the normal distribution.

    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to
        use for drawing realizations from this
        distribution. Think of it like a random seed.
        If None (or np.random), the global
        np.random state is used. If integer, it is used to
        seed the local
        :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    Distribution : Class representing general probability distributions.

    Examples
    --------
    >>> from probnum.prob.random_variable import Normal
    >>> N = Normal(mean=0.5, cov=1.0)
    >>> N.parameters
    {'mean': 0.5, 'cov': 1.0}
    """

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def __init__(
        self,
        mean: Union[float, np.floating, np.ndarray, linops.LinearOperator],
        cov: Union[float, np.floating, np.ndarray, linops.LinearOperator],
        random_state: Optional[_random_variable.RandomStateType] = None,
    ):
        # Type normalization
        if np.isscalar(mean):
            mean = _utils.as_numpy_scalar(mean)

        if np.isscalar(cov):
            cov = _utils.as_numpy_scalar(cov)

        # Shape checking
        if len(mean.shape) not in [0, 1, 2]:
            raise ValueError(
                f"Gaussian random variables must either be scalars, vectors, or "
                f"matrices (or linear operators), but the given mean is a {mean.ndim}-"
                f"dimensional tensor."
            )

        expected_cov_shape = (np.prod(mean.shape),) * 2 if len(mean.shape) > 0 else ()

        if len(cov.shape) != len(expected_cov_shape) or cov.shape != expected_cov_shape:
            raise ValueError(
                f"The covariance matrix must be of shape {expected_cov_shape}, but "
                f"shape {cov.shape} was given."
            )

        self._mean = mean
        self._cov = cov

        self.__getitem = None
        self.__reshape = None
        self.__transpose = None

        # Method selection
        properties = {}

        univariate = len(mean.shape) == 0
        dense = isinstance(mean, np.ndarray) and isinstance(cov, np.ndarray)
        operator = isinstance(mean, linops.LinearOperator) or isinstance(
            cov, linops.LinearOperator
        )

        if univariate:
            # Univariate Gaussian
            sample = self._univariate_sample
            in_support = Normal._univariate_in_support
            pdf = self._univariate_pdf
            logpdf = self._univariate_logpdf
            cdf = self._univariate_cdf
            logcdf = self._univariate_logcdf
            quantile = None  # TODO

            properties["median"] = self._mean
            properties["var"] = self._cov
            entropy = self._univariate_entropy

            self.__getitem = self._numpy_getitem
            self.__reshape = self._numpy_reshape
            self.__transpose = self._numpy_transpose
        elif dense:
            # Multi- and matrixvariate Gaussians with dense mean and covariance
            sample = self._dense_sample
            in_support = Normal._dense_in_support
            pdf = self._dense_pdf
            logpdf = self._dense_logpdf
            cdf = self._dense_cdf
            logcdf = self._dense_logcdf
            quantile = None
            entropy = self._dense_entropy

            self.__getitem = self._numpy_getitem
            self.__reshape = self._numpy_reshape
            self.__transpose = self._numpy_transpose
        elif operator:
            # Operatorvariate Gaussians
            if isinstance(cov, linops.Kronecker):
                m, n = mean.shape

                if (
                    m != cov.A.shape[0]
                    or m != cov.A.shape[1]
                    or n != cov.B.shape[0]
                    or n != cov.B.shape[1]
                ):
                    raise ValueError(
                        "Kronecker structured kernels must have factors with the same "
                        "shape as the mean."
                    )
            elif isinstance(cov, linops.SymmetricKronecker):
                m, n = mean.shape

                if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
                    raise ValueError(
                        "Normal distributions with symmetric Kronecker structured "
                        "kernels must have square mean and square kernels factors with "
                        "matching dimensions."
                    )

            in_support = None
            sample = self._operatorvariate_sample
            pdf = None
            logpdf = None
            cdf = None
            logcdf = None
            quantile = None
            entropy = None

            if isinstance(cov, linops.SymmetricKronecker) and cov._ABequal:
                sample = self._symmetric_kronecker_identical_factors_sample
        else:
            raise ValueError(
                f"Cannot instantiate normal distribution with mean of type "
                f"{mean.__class__.__name__} and kernels of type "
                f"{cov.__class__.__name__}."
            )

        properties["mode"] = self._mean
        properties["mean"] = self._mean
        properties["cov"] = self._cov

        super().__init__(
            shape=mean.shape,
            dtype=mean.dtype,
            random_state=random_state,
            parameters={"mean": self._mean, "cov": self._cov},
            sample=sample,
            in_support=in_support,
            pdf=pdf,
            logpdf=logpdf,
            cdf=cdf,
            logcdf=logcdf,
            quantile=quantile,
            entropy=entropy,
            properties=properties,
        )

    def __getitem__(self, key):
        """
        Marginalization in multi- and matrixvariate normal distributions, expressed by
        means of (advanced) indexing, masking and slicing.

        We support all modes of array indexing presented in

        https://numpy.org/doc/1.19/reference/arrays.indexing.html.

        Note that, currently, this method does not work for normal distributions other
        than the multi- and matrixvariate versions.

        Parameters
        ----------
        key : int or slice or ndarray or tuple of None, int, slice, or ndarray
            Indices, slice objects and/or boolean masks specifying which entries to keep
            while marginalizing over all other entries.
        """

        if self.__getitem is None:
            raise NotImplementedError

        return self.__getitem(key)

    def _numpy_getitem(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Select entries from mean
        mean = self._mean[key]

        # Select submatrix from covariance matrix
        cov = self._cov.reshape(self.shape + self.shape)
        cov = cov[key][tuple([slice(None)] * mean.ndim) + key]

        if mean.ndim > 0:
            cov = cov.reshape(mean.size, mean.size)

        return Normal(
            mean=mean,
            cov=cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def reshape(self, newshape):
        if self.__reshape is None:
            raise NotImplementedError

        return self.__reshape(newshape)

    def _numpy_reshape(self, newshape):
        try:
            reshaped_mean = self._mean.reshape(newshape)
        except ValueError as exc:
            raise ValueError(
                f"Cannot reshape this normal random variable to the given shape: "
                f"{newshape}"
            ) from exc

        reshaped_cov = self._cov

        if reshaped_mean.ndim > 0 and reshaped_cov.ndim == 0:
            reshaped_cov = reshaped_cov.reshape(1, 1)

        return Normal(
            mean=reshaped_mean,
            cov=reshaped_cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def transpose(self, *axes):
        if self.__transpose is None:
            raise NotImplementedError

        return self.__transpose(*axes)

    def _numpy_transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        elif (len(axes) == 1 and axes[0] is None) or len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))

        mean_t = self._mean.transpose(*axes).copy()

        # Transpose covariance
        cov_axes = axes + tuple(mean_t.ndim + axis for axis in axes)
        cov_t = self._cov.reshape(self.shape + self.shape)
        cov_t = cov_t.transpose(*cov_axes).copy()

        if mean_t.ndim > 0:
            cov_t = cov_t.reshape(mean_t.size, mean_t.size)

        return Normal(
            mean=mean_t,
            cov=cov_t,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    # Unary arithmetic operations

    def __neg__(self) -> "Normal":
        return Normal(
            mean=-self._mean,
            cov=self._cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __pos__(self) -> "Normal":
        return Normal(
            mean=+self._mean,
            cov=self._cov,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __abs__(self):  # pylint: disable=useless-super-delegation
        # TODO: Add absolute moments of normal (https://arxiv.org/pdf/1209.4340.pdf)
        return super().__abs__()

    # Binary arithmetic operations

    def _add_normal(self, other: "Normal") -> "Normal":
        if other.shape != self.shape:
            raise ValueError(
                "Addition of two normally distributed random variables is only "
                "possible if both operands have the same shape."
            )

        return Normal(
            mean=self._mean + other._mean,
            cov=self._cov + other._cov,
            random_state=_utils.derive_random_seed(
                self.random_state, other.random_state
            ),
        )

    def _add_dirac(self, dirac_rv: "probnum.prob.random_variable.Dirac") -> "Normal":
        return Normal(
            mean=self._mean + dirac_rv.support,
            cov=self._cov,
            random_state=_utils.derive_random_seed(
                self.random_state, dirac_rv.random_state
            ),
        )

    def _sub_normal(self, other: "Normal") -> "Normal":
        if other.shape != self.shape:
            raise ValueError(
                "Subtraction of two normally distributed random variables is only "
                "possible if both operands have the same shape."
            )

        return Normal(
            mean=self._mean - other._mean,
            cov=self._cov + other._cov,
            random_state=_utils.derive_random_seed(
                self.random_state, other.random_state
            ),
        )

    def _sub_dirac(self, dirac_rv: "probnum.prob.random_variable.Dirac") -> "Normal":
        return Normal(
            mean=self._mean - dirac_rv.support,
            cov=self._cov,
            random_state=_utils.derive_random_seed(
                self.random_state, dirac_rv.random_state
            ),
        )

    def _rsub_dirac(self, dirac_rv: "probnum.prob.random_variable.Dirac") -> "Normal":
        return Normal(
            mean=dirac_rv.support - self._mean,
            cov=self._cov,
            random_state=_utils.derive_random_seed(
                dirac_rv.random_state, self.random_state
            ),
        )

    def _matmul_dirac(self, dirac_rv: "probnum.prob.random_variable.Dirac") -> "Normal":
        if self.ndim == 1 or (self.ndim == 2 and self.shape[0] == 1):
            return Normal(
                mean=self._mean @ dirac_rv.support,
                cov=dirac_rv.support.T @ (self._cov @ dirac_rv.support),
                random_state=_utils.derive_random_seed(
                    self.random_state, dirac_rv.random_state
                ),
            )
        elif self.ndim == 2 and self.shape[0] > 1:
            cov_update = linops.Kronecker(
                linops.Identity(dirac_rv.shape[0]), dirac_rv.support
            )

            return Normal(
                mean=self._mean @ dirac_rv.support,
                cov=cov_update.T @ (self._cov @ cov_update),
                random_state=_utils.derive_random_seed(
                    self.random_state, dirac_rv.random_state
                ),
            )
        else:
            raise TypeError(
                "Currently, matrix multiplication is only supported for vector- and "
                "matrix-variate Gaussians."
            )

    def _rmatmul_dirac(
        self, dirac_rv: "probnum.prob.random_variable.Dirac"
    ) -> "Normal":
        if self.ndim != 1 or (self.ndim == 2 and self.shape[1] == 1):
            raise TypeError(
                "Currently, matrix multiplication is only supported for vector-variate "
                "Gaussians."
            )

        return Normal(
            mean=dirac_rv.support @ self._mean,
            cov=dirac_rv.support @ (self._cov @ dirac_rv.support.T),
            random_state=_utils.derive_random_seed(
                dirac_rv.random_state, self.random_state
            ),
        )

    def _mul_dirac(
        self, dirac_rv: "probnum.prob.random_variable.Dirac"
    ) -> Union["Normal", "probnum.prob.random_variable.Dirac"]:
        if dirac_rv.size == 1:
            return self._scale(dirac_rv.support, dirac_rv.random_state)

        return NotImplemented

    def _truediv_dirac(
        self, dirac_rv: "probnum.prob.random_variable.Dirac"
    ) -> Union["Normal", "probnum.prob.random_variable.Dirac"]:
        if dirac_rv.size == 1:
            if dirac_rv.support == 0:
                raise ZeroDivisionError

            return self._scale(1.0 / dirac_rv.support, dirac_rv.random_state)

        return NotImplemented

    def _scale(self, scalar, other_random_state=None):
        assert scalar.size == 1

        if other_random_state is None:
            derived_random_seed = _utils.derive_random_seed(self.random_state)
        else:
            derived_random_seed = _utils.derive_random_seed(
                self.random_state, other_random_state
            )

        if scalar == 0:
            return prob.random_variable.Dirac(
                support=np.zeros_like(self._mean), random_state=derived_random_seed,
            )
        else:
            return Normal(
                mean=scalar * self._mean,
                cov=(scalar ** 2) * self._cov,
                random_state=derived_random_seed,
            )

    # Univariate Gaussians
    def _univariate_sample(self, size=()) -> np.generic:
        sample = scipy.stats.norm.rvs(
            loc=self.mean, scale=self.std, size=size, random_state=self.random_state
        )

        if np.isscalar(sample):
            sample = _utils.as_numpy_scalar(sample, dtype=self.dtype)
        else:
            sample = sample.astype(self.dtype)

        return sample

    @staticmethod
    def _univariate_in_support(x) -> bool:
        return np.isfinite(x)

    def _univariate_pdf(self, x) -> float:
        return scipy.stats.norm.pdf(x, loc=self.mean, scale=self.std)

    def _univariate_logpdf(self, x) -> float:
        return scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.std)

    def _univariate_cdf(self, x) -> float:
        return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.std)

    def _univariate_logcdf(self, x) -> float:
        return scipy.stats.norm.logcdf(x, loc=self.mean, scale=self.std)

    def _univariate_entropy(self) -> float:
        return scipy.stats.norm.entropy(loc=self.mean, scale=self.std)

    # Multi- and matrixvariate Gaussians with dense covariance
    def _dense_sample(self, size=()) -> np.ndarray:
        sample = scipy.stats.multivariate_normal.rvs(
            mean=self._mean.ravel(),
            cov=self._cov,
            size=size,
            random_state=self.random_state,
        )

        return sample.reshape(sample.shape[:-1] + self.shape)

    @staticmethod
    def _dense_in_support(x) -> bool:
        return np.all(np.isfinite(x))

    def _dense_pdf(self, x) -> float:
        return scipy.stats.multivariate_normal.pdf(
            x.ravel(), mean=self._mean.ravel(), cov=self._cov
        )

    def _dense_logpdf(self, x) -> float:
        return scipy.stats.multivariate_normal.logpdf(
            x.ravel(), mean=self._mean.ravel(), cov=self._cov
        )

    def _dense_cdf(self, x) -> float:
        return scipy.stats.multivariate_normal.cdf(
            x.ravel(), mean=self._mean.ravel(), cov=self._cov
        )

    def _dense_logcdf(self, x) -> float:
        return scipy.stats.multivariate_normal.logcdf(
            x.ravel(), mean=self._mean.ravel(), cov=self._cov
        )

    def _dense_entropy(self) -> float:
        return scipy.stats.multivariate_normal.entropy(
            mean=self._mean.ravel(), cov=self._cov,
        )

    # Operatorvariate Gaussians
    def _operatorvariate_params_todense(self):
        if isinstance(self._mean, linops.LinearOperator):
            mean = self._mean.todense()
        else:
            mean = self._mean

        if isinstance(self._cov, linops.LinearOperator):
            cov = self._cov.todense()
        else:
            cov = self._cov

        return mean, cov

    def _operatorvariate_sample(self, size=()) -> np.ndarray:
        mean, cov = self._operatorvariate_params_todense()

        sample = scipy.stats.multivariate_normal.rvs(
            mean=mean.ravel(), cov=cov, size=size, random_state=self.random_state,
        )

        return sample.reshape(sample.shape[:-1] + self.shape)

    # Operatorvariate Gaussian with symmetric Kronecker covariance from identical
    # factors
    def _symmetric_kronecker_identical_factors_sample(self, size=()):
        assert isinstance(self._cov, linops.SymmetricKronecker) and self._cov._ABequal

        n = self._mean.shape[1]

        # Draw standard normal samples
        if np.isscalar(size):
            size = (size,)

        size_sample = (n * n,) + size

        stdnormal_samples = scipy.stats.norm.rvs(
            size=size_sample, random_state=self.random_state
        )

        # Cholesky decomposition
        eps = 10 ** -12  # damping needed to avoid negative definite covariances
        cholA = scipy.linalg.cholesky(
            self._cov.A.todense() + eps * np.eye(n), lower=True
        )

        # Scale and shift
        # TODO: can we avoid todense here and just return operator samples?
        if isinstance(self._mean, scipy.sparse.linalg.LinearOperator):
            mean = self._mean.todense()
        else:
            mean = self._mean

        # Appendix E: Bartels, S., Probabilistic Linear Algebra, PhD Thesis 2019
        samples_scaled = linops.Symmetrize(dim=n) @ (
            linops.Kronecker(A=cholA, B=cholA) @ stdnormal_samples
        )

        return mean[None, :, :] + samples_scaled.T.reshape(-1, n, n)
