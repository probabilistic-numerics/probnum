"""Normally distributed / Gaussian random variables."""

import functools
from typing import Callable, Optional, Union

import numpy as np
import scipy.linalg
import scipy.stats

from probnum import backend, config, linops, utils as _utils
from probnum.typing import (
    ArrayIndicesLike,
    ArrayLike,
    ArrayType,
    FloatLike,
    MatrixType,
    ShapeLike,
    ShapeType,
)

from . import _random_variable

_ValueType = Union[np.floating, np.ndarray, linops.LinearOperator]


# pylint: disable="too-complex"


class Normal(_random_variable.ContinuousRandomVariable[_ValueType]):
    """Random variable with a normal distribution.

    Gaussian random variables are ubiquitous in probability theory, since the
    Gaussian is the equilibrium distribution to which other distributions gravitate
    under a wide variety of smooth operations, e.g., convolutions and stochastic
    transformations. One example of this is the central limit theorem. Gaussian
    random variables are also attractive from a numerical point of view as they
    maintain their distribution family through many transformations (e.g. they are
    stable). In particular, they allow for efficient closed-form Bayesian inference
    given linear observations.

    Parameters
    ----------
    mean :
        Mean of the random variable.
    cov :
        (Co-)variance of the random variable.
    cov_cholesky :
        (Lower triangular) Cholesky factor of the covariance matrix. If None, then the Cholesky factor of the covariance matrix
        is computed when :attr:`Normal.cov_cholesky` is called and then cached. If specified, the value is returned by :attr:`Normal.cov_cholesky`.
        In this case, its type and data type are compared to the type and data type of the covariance.
        If the types do not match, an exception is thrown. If the data types do not match,
        the data type of the Cholesky factor is promoted to the data type of the covariance matrix.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum import randvars
    >>> x = randvars.Normal(mean=0.5, cov=1.0)
    >>> rng = np.random.default_rng(42)
    >>> x.sample(rng=rng, size=(2, 2))
    array([[ 0.80471708, -0.53998411],
           [ 1.2504512 ,  1.44056472]])
    """

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def __init__(
        self,
        mean: Union[ArrayLike, linops.LinearOperator],
        cov: Union[ArrayLike, linops.LinearOperator],
        cov_cholesky: Optional[Union[ArrayLike, linops.LinearOperator]] = None,
    ):
        # Type normalization
        if not isinstance(mean, linops.LinearOperator):
            mean = backend.asarray(mean)

        if not isinstance(cov, linops.LinearOperator):
            cov = backend.asarray(cov)

        if not isinstance(cov_cholesky, linops.LinearOperator):
            cov = backend.asarray(cov)

        # Data type normalization
        dtype = backend.promote_types(mean.dtype, cov.dtype)

        if not backend.is_floating:
            dtype = backend.double

        mean = backend.cast(mean, dtype=dtype, casting="safe", copy=False)
        cov = backend.cast(cov, dtype=dtype, casting="safe", copy=False)

        # Shape checking
        expected_cov_shape = (
            (functools.reduce(lambda a, b: a * b, mean.shape, 1),) * 2
            if mean.ndim > 0
            else ()
        )

        if cov.shape != expected_cov_shape:
            raise ValueError(
                f"The covariance matrix must be of shape {expected_cov_shape}, but "
                f"shape {cov.shape} was given."
            )

        # Method selection
        scalar = mean.ndim == 0
        dense = isinstance(mean, backend.ndarray) and isinstance(cov, backend.ndarray)
        cov_operator = isinstance(cov, linops.LinearOperator)
        compute_cov_cholesky: Callable[[], _ValueType] = None

        if scalar:
            # Scalar Gaussian
            sample = self._scalar_sample
            in_support = Normal._scalar_in_support
            pdf = self._scalar_pdf
            logpdf = self._scalar_logpdf
            cdf = self._scalar_cdf
            logcdf = self._scalar_logcdf
            quantile = self._scalar_quantile

            median = lambda: mean
            var = lambda: cov
            entropy = self._scalar_entropy

            compute_cov_cholesky = self._scalar_cov_cholesky

        elif dense or cov_operator:
            # Multi- and matrix- and tensorvariate Gaussians
            sample = self._dense_sample
            in_support = Normal._dense_in_support
            pdf = self._dense_pdf
            logpdf = self._dense_logpdf
            cdf = self._dense_cdf
            logcdf = self._dense_logcdf
            quantile = None

            median = None
            var = self._dense_var
            entropy = self._dense_entropy

            compute_cov_cholesky = self.dense_cov_cholesky

            # Ensure that the Cholesky factor has the same type as the covariance,
            # and, if necessary, promote data types. Check for (in this order): type, shape, dtype.
            if cov_cholesky is not None:
                if not isinstance(cov_cholesky, type(cov)):
                    raise TypeError(
                        f"The covariance matrix is of type `{type(cov)}`, so its "
                        f"Cholesky decomposition must be of the same type, but an "
                        f"object of type `{type(cov_cholesky)}` was given."
                    )

                if cov_cholesky.shape != cov.shape:
                    raise ValueError(
                        f"The cholesky decomposition of the covariance matrix must "
                        f"have the same shape as the covariance matrix, i.e. "
                        f"{cov.shape}, but shape {cov_cholesky.shape} was given"
                    )

                if cov_cholesky.dtype != cov.dtype:
                    cov_cholesky = backend.cast(
                        cov_cholesky, dtype=cov.dtype, casting="safe", copy=False
                    )

            if cov_operator:
                if isinstance(cov, linops.SymmetricKronecker):
                    m, n = mean.shape

                    if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
                        raise ValueError(
                            "Normal distributions with symmetric Kronecker structured "
                            "kernels must have square mean and square kernels factors with "
                            "matching dimensions."
                        )

                    if cov.identical_factors:
                        sample = self._symmetric_kronecker_identical_factors_sample

                        compute_cov_cholesky = (
                            self._symmetric_kronecker_identical_factors_cov_cholesky
                        )
                elif isinstance(cov, linops.Kronecker):
                    compute_cov_cholesky = self._kronecker_cov_cholesky
                else:
                    # This case handles all linear operators, for which no Cholesky
                    # factorization is implemented, yet.
                    # Computes the dense Cholesky and converts it to a LinearOperator.
                    compute_cov_cholesky = self._dense_cov_cholesky_as_linop
        else:
            raise ValueError(
                f"Cannot instantiate normal distribution with mean of type "
                f"{mean.__class__.__name__} and kernels of type "
                f"{cov.__class__.__name__}."
            )

        super().__init__(
            shape=mean.shape,
            dtype=mean.dtype,
            parameters={"mean": mean, "cov": cov},
            sample=sample,
            in_support=in_support,
            pdf=pdf,
            logpdf=logpdf,
            cdf=cdf,
            logcdf=logcdf,
            quantile=quantile,
            mode=lambda: mean,
            median=median,
            mean=lambda: mean,
            cov=lambda: cov,
            var=var,
            entropy=entropy,
        )

        self._compute_cov_cholesky = compute_cov_cholesky
        self._cov_cholesky = cov_cholesky

    @property
    def cov_cholesky(self) -> _ValueType:
        """Cholesky factor :math:`L` of the covariance
        :math:`\\operatorname{Cov}(X) =LL^\\top`."""

        if not self.cov_cholesky_is_precomputed:
            self.precompute_cov_cholesky()
        return self._cov_cholesky

    def precompute_cov_cholesky(
        self,
        damping_factor: Optional[FloatLike] = None,
    ):
        """(P)recompute Cholesky factors (careful: in-place operation!)."""
        if damping_factor is None:
            damping_factor = config.covariance_inversion_damping
        if self.cov_cholesky_is_precomputed:
            raise Exception("A Cholesky factor is already available.")
        self._cov_cholesky = self._compute_cov_cholesky(damping_factor=damping_factor)

    @property
    def cov_cholesky_is_precomputed(self) -> bool:
        """Return truth-value of whether the Cholesky factor of the covariance is
        readily available.

        This happens if (i) the Cholesky factor is specified during
        initialization or if (ii) the property `self.cov_cholesky` has
        been called before.
        """
        if self._cov_cholesky is None:
            return False
        return True

    @functools.cached_property
    def dense_mean(self) -> Union[np.floating, np.ndarray]:
        """Dense representation of the mean."""
        if isinstance(self.mean, linops.LinearOperator):
            return self.mean.todense()
        else:
            return self.mean

    @functools.cached_property
    def dense_cov(self) -> Union[np.floating, np.ndarray]:
        """Dense representation of the covariance."""
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov.todense()
        else:
            return self.cov

    def __getitem__(self, key: ArrayIndicesLike) -> "Normal":
        """Marginalization in multi- and matrixvariate normal random variables,
        expressed as (advanced) indexing, masking and slicing.

        We support all modes of array indexing presented in

        https://numpy.org/doc/1.19/reference/arrays.indexing.html.

        Note that, currently, this method only works for multi- and matrixvariate
        normal distributions.

        Parameters
        ----------
        key : int or slice or ndarray or tuple of None, int, slice, or ndarray
            Indices, slice objects and/or boolean masks specifying which entries to keep
            while marginalizing over all other entries.
        """

        if not isinstance(key, tuple):
            key = (key,)

        # Select entries from mean
        mean = self.dense_mean[key]

        # Select submatrix from covariance matrix
        cov = self.dense_cov.reshape(self.shape + self.shape)
        cov = cov[key][(...,) + key]

        if mean.ndim > 0:
            cov = cov.reshape(mean.size, mean.size)

        return Normal(
            mean=mean,
            cov=cov,
        )

    def reshape(self, newshape: ShapeLike) -> "Normal":
        try:
            reshaped_mean = self.dense_mean.reshape(newshape)
        except ValueError as exc:
            raise ValueError(
                f"Cannot reshape this normal random variable to the given shape: "
                f"{newshape}"
            ) from exc

        reshaped_cov = self.dense_cov

        if reshaped_mean.ndim > 0 and reshaped_cov.ndim == 0:
            reshaped_cov = reshaped_cov.reshape(1, 1)

        return Normal(
            mean=reshaped_mean,
            cov=reshaped_cov,
        )

    def transpose(self, *axes: int) -> "Normal":
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        elif (len(axes) == 1 and axes[0] is None) or len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))

        mean_t = self.dense_mean.transpose(*axes).copy()

        # Transpose covariance
        cov_axes = axes + tuple(mean_t.ndim + axis for axis in axes)
        cov_t = self.dense_cov.reshape(self.shape + self.shape)
        cov_t = cov_t.transpose(*cov_axes).copy()

        if mean_t.ndim > 0:
            cov_t = cov_t.reshape(mean_t.size, mean_t.size)

        return Normal(
            mean=mean_t,
            cov=cov_t,
        )

    # Unary arithmetic operations

    def __neg__(self) -> "Normal":
        return Normal(
            mean=-self.mean,
            cov=self.cov,
        )

    def __pos__(self) -> "Normal":
        return Normal(
            mean=+self.mean,
            cov=self.cov,
        )

    # TODO: Overwrite __abs__ and add absolute moments of normal
    # TODO: (https://arxiv.org/pdf/1209.4340.pdf)

    # Binary arithmetic operations

    def _add_normal(self, other: "Normal") -> "Normal":
        if other.shape != self.shape:
            raise ValueError(
                "Addition of two normally distributed random variables is only "
                "possible if both operands have the same shape."
            )

        return Normal(
            mean=self.mean + other.mean,
            cov=self.cov + other.cov,
        )

    def _sub_normal(self, other: "Normal") -> "Normal":
        if other.shape != self.shape:
            raise ValueError(
                "Subtraction of two normally distributed random variables is only "
                "possible if both operands have the same shape."
            )

        return Normal(
            mean=self.mean - other.mean,
            cov=self.cov + other.cov,
        )

    # Univariate Gaussians
    def _scalar_cov_cholesky(
        self,
        damping_factor: FloatLike,
    ) -> np.floating:
        return np.sqrt(self.cov + damping_factor)

    def _scalar_sample(
        self,
        rng: np.random.Generator,
        size: ShapeType = (),
    ) -> Union[np.floating, np.ndarray]:
        sample = scipy.stats.norm.rvs(
            loc=self.mean, scale=self.std, size=size, random_state=rng
        )

        if np.isscalar(sample):
            sample = _utils.as_numpy_scalar(sample, dtype=self.dtype)
        else:
            sample = sample.astype(self.dtype)

        assert sample.shape == size

        return sample

    @staticmethod
    def _scalar_in_support(x: _ValueType) -> bool:
        return np.isfinite(x)

    @backend.jit_method
    def _scalar_pdf(self, x: _ValueType) -> np.float_:
        return backend.exp(-((x - self.mean) ** 2) / (2.0 * self.var)) / backend.sqrt(
            2 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_logpdf(self, x: _ValueType) -> ArrayType:
        return -((x - self.mean) ** 2) / (2.0 * self.var) - 0.5 * backend.log(
            2.0 * backend.pi * self.var
        )

    def _scalar_cdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.std)

    def _scalar_logcdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.norm.logcdf(x, loc=self.mean, scale=self.std)

    def _scalar_quantile(self, p: FloatLike) -> np.floating:
        return scipy.stats.norm.ppf(p, loc=self.mean, scale=self.std)

    def _scalar_entropy(self: _ValueType) -> np.float_:
        return _utils.as_numpy_scalar(
            scipy.stats.norm.entropy(loc=self.mean, scale=self.std),
            dtype=np.float_,
        )

    # Multi- and matrixvariate Gaussians
    def dense_cov_cholesky(
        self,
        damping_factor: Optional[FloatLike] = None,
    ) -> np.ndarray:
        """Compute the Cholesky factorization of the covariance from its dense
        representation."""
        if damping_factor is None:
            damping_factor = config.covariance_inversion_damping
        dense_cov = self.dense_cov

        return backend.linalg.cholesky(
            dense_cov + damping_factor * backend.eye(self.size, dtype=self.dtype),
            lower=True,
        )

    def _dense_cov_cholesky_as_linop(
        self, damping_factor: FloatLike
    ) -> linops.LinearOperator:
        return linops.aslinop(self.dense_cov_cholesky(damping_factor=damping_factor))

    def _dense_sample(
        self, rng: np.random.Generator, size: ShapeType = ()
    ) -> np.ndarray:
        sample = scipy.stats.multivariate_normal.rvs(
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
            size=size,
            random_state=rng,
        )

        return sample.reshape(sample.shape[:-1] + self.shape)

    @staticmethod
    def _arg_todense(x: Union[np.ndarray, linops.LinearOperator]) -> np.ndarray:
        if isinstance(x, linops.LinearOperator):
            return x.todense()
        elif isinstance(x, backend.ndarray):
            return x
        else:
            raise ValueError(f"Unsupported argument type {type(x)}")

    @staticmethod
    def _dense_in_support(x: _ValueType) -> bool:
        return np.all(np.isfinite(Normal._arg_todense(x)))

    def _dense_pdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.pdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_logpdf(self, x: _ValueType) -> ArrayType:
        x_centered = Normal._arg_todense(x - self.dense_mean).reshape(
            x.shape[: -self.ndim] + (-1,)
        )

        return -0.5 * (
            x_centered @ backend.linalg.cho_solve((self.cov_cholesky, True), x_centered)
            + self.size * backend.log(backend.array(2.0 * backend.pi))
        ) - backend.sum(backend.log(backend.diag(self.cov_cholesky)))

    def _dense_cdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.cdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_logcdf(self, x: _ValueType) -> np.float_:
        return scipy.stats.multivariate_normal.logcdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _dense_var(self) -> np.ndarray:
        return np.diag(self.dense_cov).reshape(self.shape)

    def _dense_entropy(self) -> np.float_:
        return _utils.as_numpy_scalar(
            scipy.stats.multivariate_normal.entropy(
                mean=self.dense_mean.ravel(),
                cov=self.dense_cov,
            ),
            dtype=np.float_,
        )

    # Matrixvariate Gaussian with Kronecker covariance
    def _kronecker_cov_cholesky(
        self,
        damping_factor: FloatLike,
    ) -> linops.Kronecker:
        assert isinstance(self.cov, linops.Kronecker)

        A = self.cov.A.todense()
        B = self.cov.B.todense()

        return linops.Kronecker(
            A=scipy.linalg.cholesky(
                A + damping_factor * np.eye(A.shape[0], dtype=self.dtype),
                lower=True,
            ),
            B=scipy.linalg.cholesky(
                B + damping_factor * np.eye(B.shape[0], dtype=self.dtype),
                lower=True,
            ),
        )

    # Matrixvariate Gaussian with symmetric Kronecker covariance from identical
    # factors
    def _symmetric_kronecker_identical_factors_cov_cholesky(
        self,
        damping_factor: FloatLike,
    ) -> linops.SymmetricKronecker:
        assert (
            isinstance(self.cov, linops.SymmetricKronecker)
            and self.cov.identical_factors
        )

        A = self.cov.A.todense()

        return linops.SymmetricKronecker(
            A=scipy.linalg.cholesky(
                A + damping_factor * np.eye(A.shape[0], dtype=self.dtype),
                lower=True,
            ),
        )

    def _symmetric_kronecker_identical_factors_sample(
        self, rng: np.random.Generator, size: ShapeType = ()
    ) -> np.ndarray:
        assert (
            isinstance(self.cov, linops.SymmetricKronecker)
            and self.cov.identical_factors
        )

        n = self.mean.shape[1]

        # Draw standard normal samples
        size_sample = (n * n,) + size

        stdnormal_samples = scipy.stats.norm.rvs(size=size_sample, random_state=rng)

        # Appendix E: Bartels, S., Probabilistic Linear Algebra, PhD Thesis 2019
        samples_scaled = linops.Symmetrize(n) @ (self.cov_cholesky @ stdnormal_samples)

        # TODO: can we avoid todense here and just return operator samples?
        return self.dense_mean[None, :, :] + samples_scaled.T.reshape(-1, n, n)
