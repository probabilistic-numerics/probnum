"""Normally distributed / Gaussian random variables."""
from __future__ import annotations

import functools
import operator
from typing import Optional, Union

from probnum import backend, linops
from probnum.typing import (
    ArrayIndicesLike,
    ArrayLike,
    FloatLike,
    MatrixType,
    ScalarType,
    SeedLike,
    SeedType,
    ShapeLike,
    ShapeType,
)

from . import _random_variable


class Normal(_random_variable.ContinuousRandomVariable):
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
        (Lower triangular) Cholesky factor of the covariance matrix. If ``None``, then
        the Cholesky factor of the covariance matrix is computed when
        :attr:`Normal.cov_cholesky` is called and then cached. If specified, the value
        is returned by :attr:`Normal.cov_cholesky`. In this case, its type and data type
        are compared to the type and data type of the covariance. If the types do not
        match, an exception is thrown. If the data types do not match, the data type of
        the Cholesky factor is promoted to the data type of the covariance matrix.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Examples
    --------
    >>> x = pn.randvars.Normal(mean=0.5, cov=1.0)
    >>> rng = np.random.default_rng(42)
    >>> x.sample(rng=rng, size=(2, 2))
    array([[ 0.80471708, -0.53998411],
           [ 1.2504512 ,  1.44056472]])
    """

    # TODO (#xyz): `cov_cholesky` should be passed to the `cov` `LinearOperator`
    def __init__(
        self,
        mean: Union[ArrayLike, linops.LinearOperator],
        cov: Union[ArrayLike, linops.LinearOperator],
        cov_cholesky: Optional[Union[ArrayLike, linops.LinearOperator]] = None,
    ):
        # pylint: disable=too-many-branches

        # Type normalization
        if not isinstance(mean, linops.LinearOperator):
            mean = backend.asarray(mean)

        if not isinstance(cov, linops.LinearOperator):
            cov = backend.asarray(cov)

        # Data type normalization
        dtype = backend.promote_types(mean.dtype, cov.dtype)

        if not backend.is_floating_dtype(dtype):
            dtype = backend.double

        # Circular dependency -> defer import
        from probnum import compat  # pylint: disable=import-outside-toplevel

        mean = compat.cast(mean, dtype=dtype, casting="safe", copy=False)
        cov = compat.cast(cov, dtype=dtype, casting="safe", copy=False)

        if cov_cholesky is not None:
            cov_cholesky = compat.cast(cov_cholesky, dtype, copy=False)

        # Shape checking
        expected_cov_shape = (
            (functools.reduce(operator.mul, mean.shape, 1),) * 2
            if mean.ndim > 0
            else ()
        )

        if cov.shape != expected_cov_shape:
            raise ValueError(
                f"The covariance matrix must be of shape {expected_cov_shape}, but "
                f"shape {cov.shape} was given."
            )

        if cov_cholesky is not None:
            if cov_cholesky.shape != cov.shape:
                raise ValueError(
                    f"The Cholesky decomposition of the covariance matrix must "
                    f"have the same shape as the covariance matrix, i.e. "
                    f"{cov.shape}, but shape {cov_cholesky.shape} was given"
                )

        self._cov_cholesky = cov_cholesky
        self.__cov_eigh = None

        if mean.ndim == 0:
            # Scalar Gaussian
            super().__init__(
                shape=(),
                dtype=mean.dtype,
                parameters={"mean": mean, "cov": cov},
                sample=self._scalar_sample,
                in_support=Normal._scalar_in_support,
                pdf=self._scalar_pdf,
                logpdf=self._scalar_logpdf,
                cdf=self._scalar_cdf,
                logcdf=self._scalar_logcdf,
                quantile=self._scalar_quantile,
                mode=lambda: mean,
                median=lambda: mean,
                mean=lambda: mean,
                cov=lambda: cov,
                var=lambda: cov,
                entropy=self._scalar_entropy,
            )
        else:
            # Multi- and matrix- and tensorvariate Gaussians
            super().__init__(
                shape=mean.shape,
                dtype=mean.dtype,
                parameters={"mean": mean, "cov": cov},
                sample=self._sample,
                in_support=self._in_support,
                pdf=self._pdf,
                logpdf=self._logpdf,
                cdf=self._cdf,
                logcdf=self._logcdf,
                quantile=None,
                mode=lambda: mean,
                median=None,
                mean=lambda: mean,
                cov=lambda: cov,
                var=self._var,
                entropy=self._entropy,
            )

    @property
    def dense_mean(self) -> backend.ndarray:
        """Dense representation of the mean."""
        if isinstance(self.mean, linops.LinearOperator):
            return self.mean.todense()

        return self.mean

    @property
    def dense_cov(self) -> backend.ndarray:
        """Dense representation of the covariance."""
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov.todense()

        return self.cov

    @functools.cached_property
    def cov_matrix(self) -> backend.ndarray:
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov.todense()

        return self.cov

    @functools.cached_property
    def cov_op(self) -> linops.LinearOperator:
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov

        return linops.aslinop(self.cov)

    # TODO (#xyz): Use `LinearOperator.cholesky` once the backend is supported

    @property
    def cov_cholesky(self) -> MatrixType:
        """Cholesky factor :math:`L` of the covariance
        :math:`\\operatorname{Cov}(X) =LL^\\top`."""

        if not self.cov_cholesky_is_precomputed:
            self.compute_cov_cholesky()

        return self._cov_cholesky

    @functools.cached_property
    def _cov_matrix_cholesky(self) -> backend.ndarray:
        if isinstance(self._cov_cholesky, linops.LinearOperator):
            return self._cov_cholesky.todense()

        return self._cov_cholesky

    @functools.cached_property
    def _cov_op_cholesky(self) -> linops.LinearOperator:
        if isinstance(self._cov_cholesky, backend.ndarray):
            return linops.aslinop(self._cov_cholesky)

        return self._cov_cholesky

    def compute_cov_cholesky(self) -> None:
        """Compute Cholesky factor (careful: in-place operation!)."""

        if self.cov_cholesky_is_precomputed:
            raise Exception("A Cholesky factor is already available.")

        if self.ndim == 0:
            self._cov_cholesky = backend.sqrt(self.cov)
        elif isinstance(self.cov, backend.ndarray):
            self._cov_cholesky = backend.linalg.cholesky(self.cov, lower=True)
        else:
            assert isinstance(self.cov, linops.LinearOperator)

            self._cov_cholesky = self.cov.cholesky(lower=True)

    @property
    def cov_cholesky_is_precomputed(self) -> bool:
        """Return truth-value of whether the Cholesky factor of the covariance is
        readily available.

        This happens if (i) the Cholesky factor is specified during initialization or if
        (ii) the property `self.cov_cholesky` has been called before.
        """
        return self._cov_cholesky is not None

    # TODO (#xyz): Use `LinearOperator.eig` once the backend is supported

    @property
    def _cov_eigh(self):
        return self.__cov_eigh

    def compute_cov_eigh(self) -> None:
        if self.cov_eigh_is_precomputed:
            raise Exception("An eigendecomposition is already available.")

        if self.ndim == 0:
            eigvals = self.cov
            Q = backend.ones_like(self.cov)
        elif isinstance(self.cov, backend.ndarray):
            eigvals, Q = backend.linalg.eigh(self.cov)
        elif isinstance(self.cov, linops.Kronecker):
            A_eigvals, A_eigvecs = backend.linalg.eigh(self.cov.A.todense())
            B_eigvals, B_eigvecs = backend.linalg.eigh(self.cov.B.todense())

            eigvals = backend.kron(A_eigvals, B_eigvals)
            Q = linops.Kronecker(A_eigvecs, B_eigvecs)
        elif (
            isinstance(self.cov, linops.SymmetricKronecker)
            and self.cov.identical_factors
        ):
            A_eigvals, A_eigvecs = backend.linalg.eigh(self.cov.A.todense())

            eigvals = backend.kron(A_eigvals, B_eigvals)
            Q = linops.SymmetricKronecker(A_eigvecs)
        else:
            assert isinstance(self.cov, linops.LinearOperator)

            eigvals, Q = backend.linalg.eigh(self.dense_cov)

            Q = linops.aslinop(Q)

        # Clip eigenvalues as in
        # https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/stats/_multivariate.py#L60-L166
        if self.dtype == backend.double:
            eigvals_clip = 1e6
        elif self.dtype == backend.single:
            eigvals_clip = 1e3
        else:
            raise TypeError("Unsupported dtype")

        eigvals_clip *= backend.finfo(self.dtype).eps
        eigvals_clip *= backend.max(backend.abs(eigvals))

        if backend.any(eigvals < -eigvals_clip):
            raise backend.linalg.LinAlgError(
                "The covariance matrix is not positive semi-definite."
            )

        eigvals = eigvals * (eigvals >= eigvals_clip)

        self._cov_eigh = (eigvals, Q)

    @property
    def cov_eigh_is_precomputed(self) -> bool:
        return self.__cov_eigh is not None

    @functools.cached_property
    def _cov_sqrtm(self) -> MatrixType:
        if not self.cov_eigh_is_precomputed:
            # Attempt Cholesky factorization
            try:
                return self.cov_cholesky
            except backend.linalg.LinAlgError:
                pass

        # Fall back to symmetric eigendecomposition
        eigvals, Q = self._cov_eigh

        if isinstance(Q, linops.LinearOperator):
            return Q @ linops.Scaling(backend.sqrt(eigvals))

        return Q * backend.sqrt(eigvals)[None, :]

    def _cov_sqrtm_solve(self, x: backend.ndarray) -> backend.ndarray:
        if not self.cov_eigh_is_precomputed:
            # Attempt Cholesky factorization
            try:
                cov_matrix_cholesky = self._cov_matrix_cholesky
            except backend.linalg.LinAlgError:
                cov_matrix_cholesky = None

            if cov_matrix_cholesky is not None:
                return backend.linalg.solve_triangular(
                    self._cov_matrix_cholesky,
                    x[..., None],
                    lower=True,
                )[..., 0]

        # Fall back to symmetric eigendecomposition
        eigvals, Q = self._cov_eigh

        return (x @ Q) / backend.sqrt(eigvals)

    @functools.cached_property
    def _cov_logdet(self) -> backend.ndarray:
        if not self.cov_eigh_is_precomputed:
            # Attempt Cholesky factorization
            try:
                cov_matrix_cholesky = self._cov_matrix_cholesky
            except backend.linalg.LinAlgError:
                cov_matrix_cholesky = None

            if cov_matrix_cholesky is not None:
                return 2.0 * backend.sum(backend.log(backend.diag(cov_matrix_cholesky)))

        # Fall back to symmetric eigendecomposition
        eigvals, _ = self._cov_eigh

        return backend.sum(backend.log(eigvals))

    def __getitem__(self, key: ArrayIndicesLike) -> "Normal":
        """Marginalization in multi- and matrixvariate normal random variables,
        expressed as (advanced) indexing, masking and slicing.

        We support all modes of array indexing presented in

        https://numpy.org/doc/1.19/reference/arrays.indexing.html.

        Parameters
        ----------
        key :
            Indices, slice objects and/or boolean masks specifying which entries to keep
            while marginalizing over all other entries.

        Returns
        -------
        Random variable after marginalization.
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
    @functools.partial(backend.jit_method, static_argnums=(1,))
    def _scalar_sample(
        self,
        seed: SeedType,
        sample_shape: ShapeType = (),
    ) -> backend.ndarray:
        sample = backend.random.standard_normal(
            seed,
            shape=sample_shape,
            dtype=self.dtype,
        )

        return self.std * sample + self.mean

    @staticmethod
    @backend.jit
    def _scalar_in_support(x: backend.ndarray) -> backend.ndarray:
        return backend.isfinite(x)

    @backend.jit_method
    def _scalar_pdf(self, x: backend.ndarray) -> backend.ndarray:
        return backend.exp(-((x - self.mean) ** 2) / (2.0 * self.var)) / backend.sqrt(
            2 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_logpdf(self, x: backend.ndarray) -> backend.ndarray:
        return -((x - self.mean) ** 2) / (2.0 * self.var) - 0.5 * backend.log(
            2.0 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_cdf(self, x: backend.ndarray) -> backend.ndarray:
        return backend.special.ndtr((x - self.mean) / self.std)

    @backend.jit_method
    def _scalar_logcdf(self, x: backend.ndarray) -> backend.ndarray:
        return backend.log(self._scalar_cdf(x))

    @backend.jit_method
    def _scalar_quantile(self, p: FloatLike) -> backend.ndarray:
        return self.mean + self.std * backend.special.ndtri(p)

    @backend.jit_method
    def _scalar_entropy(self) -> ScalarType:
        return 0.5 * backend.log(2.0 * backend.pi * self.var) + 0.5

    # Multi- and matrixvariate Gaussians

    # TODO (#xyz): jit this function once `LinearOperator`s support the backend
    # @functools.partial(backend.jit_method, static_argnums=(1,))
    def _sample(self, seed: SeedLike, sample_shape: ShapeType = ()) -> backend.ndarray:
        samples = backend.random.standard_normal(
            seed,
            shape=sample_shape + (self.size,),
            dtype=self.dtype,
        )

        samples = backend.asarray((self._cov_sqrtm @ samples[..., None])[..., 0])
        samples += self.dense_mean

        return samples.reshape(sample_shape + self.shape)

    @staticmethod
    def _arg_todense(
        x: Union[backend.ndarray, linops.LinearOperator]
    ) -> backend.ndarray:
        if isinstance(x, linops.LinearOperator):
            return x.todense()

        if isinstance(x, backend.ndarray):
            return x

        raise ValueError(f"Unsupported argument type {type(x)}")

    @backend.jit_method
    def _in_support(self, x: backend.ndarray) -> backend.ndarray:
        return backend.all(
            backend.isfinite(Normal._arg_todense(x)),
            axis=tuple(range(-self.ndim, 0)),
            keepdims=False,
        )

    @backend.jit_method
    def _pdf(self, x: backend.ndarray) -> backend.ndarray:
        return backend.exp(self._logpdf(x))

    @backend.jit_method
    def _logpdf(self, x: backend.ndarray) -> backend.ndarray:
        x_centered = Normal._arg_todense(x - self.dense_mean).reshape(
            x.shape[: -self.ndim] + (-1,)
        )

        return -0.5 * (
            # TODO (#xyz): backend.sum(
            #     x_centered * self._cov_op.inv()(x_centered, axis=-1),
            #     axis=-1
            # )
            # Here, we use:
            # ||L^{-1}(x - \mu)||_2^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)
            backend.sum(self._cov_sqrtm_solve(x_centered) ** 2, axis=-1)
            + self.size * backend.log(backend.array(2.0 * backend.pi))
            # TODO (#xyz): Replace this with `self._cov_op.logdet()`
            + self._cov_logdet
        )

    _cdf = backend.Dispatcher()

    @_cdf.numpy
    def _cdf_numpy(self, x: backend.ndarray) -> backend.ndarray:
        import scipy.stats  # pylint: disable=import-outside-toplevel

        scipy_cdf = scipy.stats.multivariate_normal.cdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

        # scipy's implementation happily squeezes `1` dimensions out of the batch
        expected_shape = x.shape[: x.ndim - self.ndim]

        if any(dim == 1 for dim in expected_shape):
            assert all(dim != 1 for dim in scipy_cdf.shape)

            scipy_cdf = scipy_cdf.reshape(expected_shape)

        return scipy_cdf

    def _logcdf(self, x: backend.ndarray) -> backend.ndarray:
        return backend.log(self.cdf(x))

    @backend.jit_method
    def _var(self) -> backend.ndarray:
        return backend.diag(self.dense_cov).reshape(self.shape)

    @backend.jit_method
    def _entropy(self) -> ScalarType:
        entropy = 0.5 * self.size * (backend.log(2.0 * backend.pi) + 1.0)
        # TODO (#xyz): Replace this with `0.5 * self._cov_op.logdet()`
        entropy += 0.5 * self._cov_logdet

        return entropy
