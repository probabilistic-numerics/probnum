"""Normally distributed / Gaussian random variables."""
from __future__ import annotations

import functools
import operator
from typing import Any, Dict, Optional, Union

from probnum import backend, linops
from probnum.backend.typing import (
    ArrayIndicesLike,
    ArrayLike,
    FloatLike,
    SeedLike,
    SeedType,
    ShapeLike,
    ShapeType,
)
from probnum.typing import MatrixType

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
        :attr:`Normal._cov_cholesky` is called and then cached. If specified, the value
        is returned by :attr:`Normal._cov_cholesky`. In this case, its type and data
        type are compared to the type and data type of the covariance. If the types do
        not match, an exception is thrown. If the data types do not match, the data type
        of the Cholesky factor is promoted to the data type of the covariance matrix.

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

    # TODO (#678): `cov_cholesky` should be passed to the `cov` `LinearOperator`
    def __init__(
        self,
        mean: Union[ArrayLike, linops.LinearOperator],
        cov: Union[ArrayLike, linops.LinearOperator],
        cache: Optional[Dict[str, Any]] = None,
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

        self._cache = cache if cache is not None else {}

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
    def dense_mean(self) -> backend.Array:
        """Dense representation of the mean."""
        if isinstance(self.mean, linops.LinearOperator):
            return self.mean.todense()

        return self.mean

    @property
    def dense_cov(self) -> backend.Array:
        """Dense representation of the covariance."""
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov.todense()

        return self.cov

    @functools.cached_property
    def cov_matrix(self) -> backend.Array:
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov.todense()

        return self.cov

    @functools.cached_property
    def cov_op(self) -> linops.LinearOperator:
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov

        return linops.aslinop(self.cov)

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
    ) -> backend.Array:
        sample = backend.random.standard_normal(
            seed,
            shape=sample_shape,
            dtype=self.dtype,
        )

        return self.std * sample + self.mean

    @staticmethod
    @backend.jit
    def _scalar_in_support(x: backend.Array) -> backend.Array:
        return backend.isfinite(x)

    @backend.jit_method
    def _scalar_pdf(self, x: backend.Array) -> backend.Array:
        return backend.exp(-((x - self.mean) ** 2) / (2.0 * self.var)) / backend.sqrt(
            2 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_logpdf(self, x: backend.Array) -> backend.Array:
        return -((x - self.mean) ** 2) / (2.0 * self.var) - 0.5 * backend.log(
            2.0 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_cdf(self, x: backend.Array) -> backend.Array:
        return backend.special.ndtr((x - self.mean) / self.std)

    @backend.jit_method
    def _scalar_logcdf(self, x: backend.Array) -> backend.Array:
        return backend.log(self._scalar_cdf(x))

    @backend.jit_method
    def _scalar_quantile(self, p: FloatLike) -> backend.Array:
        return self.mean + self.std * backend.special.ndtri(p)

    @backend.jit_method
    def _scalar_entropy(self) -> backend.Scalar:
        return 0.5 * backend.log(2.0 * backend.pi * self.var) + 0.5

    # Multi- and matrixvariate Gaussians

    # TODO (#569,#678): jit this function once `LinearOperator`s support the backend
    # @functools.partial(backend.jit_method, static_argnums=(1,))
    def _sample(self, seed: SeedLike, sample_shape: ShapeType = ()) -> backend.Array:
        samples = backend.random.standard_normal(
            seed,
            shape=sample_shape + (self.size,),
            dtype=self.dtype,
        )

        samples = backend.asarray((self._cov_sqrtm @ samples[..., None])[..., 0])
        samples += self.dense_mean

        return samples.reshape(sample_shape + self.shape)

    @staticmethod
    def _arg_todense(x: Union[backend.Array, linops.LinearOperator]) -> backend.Array:
        if isinstance(x, linops.LinearOperator):
            return x.todense()

        if backend.isarray(x):
            return x

        raise ValueError(f"Unsupported argument type {type(x)}")

    @backend.jit_method
    def _in_support(self, x: backend.Array) -> backend.Array:
        return backend.all(
            backend.isfinite(Normal._arg_todense(x)),
            axis=tuple(range(-self.ndim, 0)),
            keepdims=False,
        )

    @backend.jit_method
    def _pdf(self, x: backend.Array) -> backend.Array:
        return backend.exp(self._logpdf(x))

    @backend.jit_method
    def _logpdf(self, x: backend.Array) -> backend.Array:
        x_centered = Normal._arg_todense(x - self.dense_mean).reshape(
            x.shape[: -self.ndim] + (-1,)
        )

        return -0.5 * (
            # TODO (#569,#678): backend.sum(
            #     x_centered * self._cov_op.inv()(x_centered, axis=-1),
            #     axis=-1
            # )
            # Here, we use:
            # ||L^{-1}(x - \mu)||_2^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)
            backend.sum(self._cov_sqrtm_solve(x_centered) ** 2, axis=-1)
            + self.size * backend.log(backend.array(2.0 * backend.pi))
            + self._cov_logdet
        )

    _cdf = backend.Dispatcher()

    @_cdf.numpy_impl
    def _cdf_numpy(self, x: backend.Array) -> backend.Array:
        import scipy.stats  # pylint: disable=import-outside-toplevel

        scipy_cdf = scipy.stats.multivariate_normal.cdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.cov_matrix,
        )

        # scipy's implementation happily squeezes `1` dimensions out of the batch
        expected_shape = x.shape[: x.ndim - self.ndim]

        if any(dim == 1 for dim in expected_shape):
            assert all(dim != 1 for dim in scipy_cdf.shape)

            scipy_cdf = scipy_cdf.reshape(expected_shape)

        return scipy_cdf

    def _logcdf(self, x: backend.Array) -> backend.Array:
        return backend.log(self.cdf(x))

    @backend.jit_method
    def _var(self) -> backend.Array:
        return backend.diag(self.dense_cov).reshape(self.shape)

    @backend.jit_method
    def _entropy(self) -> backend.Scalar:
        entropy = 0.5 * self.size * (backend.log(2.0 * backend.pi) + 1.0)
        entropy += 0.5 * self._cov_logdet

        return entropy

    def compute_cov_sqrtm(self) -> Normal:
        if "cov_cholesky" in self._cache and "cov_eigh" in self._cache:
            return self

        cache = self._cache

        if "cov_cholesky" not in self._cache:
            cache["cov_cholesky"] = self._cov_cholesky

        return Normal(
            self.mean,
            self.cov,
            cache=backend.cond(
                backend.any(backend.isnan(cache["cov_cholesky"])),
                lambda: cache + {"cov_eigh": self._cov_eigh},
                lambda: cache,
            ),
        )

    # TODO (#678): Use `LinearOperator.cholesky` once the backend is supported

    @property
    @backend.jit_method
    def _cov_cholesky(self) -> MatrixType:
        if "cov_cholesky" in self._cache:
            return self._cache["cov_cholesky"]

        if self.ndim == 0:
            return backend.sqrt(self.cov)

        if backend.isarray(self.cov):
            return backend.linalg.cholesky(self.cov, upper=False)

        assert isinstance(self.cov, linops.LinearOperator)

        return self.cov.cholesky(lower=True)

    @property
    def _cov_matrix_cholesky(self) -> backend.Array:
        if isinstance(self._cov_cholesky, linops.LinearOperator):
            return self._cov_cholesky.todense()

        return self._cov_cholesky

    # TODO (#569,#678): Use `LinearOperator.eig` it is implemented and once the backend
    # is supported

    @property
    @backend.jit_method
    def _cov_eigh(self) -> MatrixType:
        if "cov_eigh" in self._cache:
            return self._cache["cov_eigh"]

        if self.ndim == 0:
            eigvals = self.cov
            Q = backend.ones_like(self.cov)
        elif backend.isarray(self.cov):
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

            eigvals, Q = backend.linalg.eigh(self.cov_matrix)

            Q = linops.aslinop(Q)

        return (_clip_eigvals(eigvals), Q)

    # TODO (#569,#678): Replace `_cov_{sqrtm,sqrtm_solve,logdet}` with
    # `self._cov_op.{sqrtm,inv,logdet}` once they are supported and once linops support
    # the backend

    @property
    @backend.jit_method
    def _cov_sqrtm(self) -> MatrixType:
        cov_cholesky = self._cov_cholesky

        def _fallback_eigh():
            eigvals, Q = self._cov_eigh

            if isinstance(Q, linops.LinearOperator):
                return Q @ linops.Scaling(backend.sqrt(eigvals))

            return Q * backend.sqrt(eigvals)[None, :]

        return backend.cond(
            backend.any(backend.isnan(cov_cholesky)),
            _fallback_eigh,
            lambda: cov_cholesky,
        )

    @backend.jit_method
    def _cov_sqrtm_solve(self, x: backend.Array) -> backend.Array:
        def _eigh_fallback(x):
            eigvals, Q = self._cov_eigh

            return (x @ Q) / backend.sqrt(eigvals)

        return backend.cond(
            backend.any(backend.isnan(self._cov_cholesky)),
            _eigh_fallback,
            lambda x: backend.linalg.solve_triangular(
                self._cov_matrix_cholesky,
                x[..., None],
                lower=True,
            )[..., 0],
            x,
        )

    @property
    @backend.jit_method
    def _cov_logdet(self) -> backend.Array:
        return backend.cond(
            backend.any(backend.isnan(self._cov_cholesky)),
            lambda: backend.sum(backend.log(self._cov_eigh[0])),
            lambda: (
                2.0 * backend.sum(backend.log(backend.diag(self._cov_matrix_cholesky)))
            ),
        )


def _clip_eigvals(eigvals: backend.Array) -> backend.Array:
    # Clip eigenvalues as in
    # https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/stats/_multivariate.py#L60-L166
    if eigvals.dtype == backend.double:
        eigvals_clip = 1e6
    elif eigvals.dtype == backend.single:
        eigvals_clip = 1e3
    else:
        raise TypeError("Unsupported dtype")

    eigvals_clip *= backend.finfo(eigvals.dtype).eps
    eigvals_clip *= backend.max(backend.abs(eigvals))

    return backend.cond(
        backend.any(eigvals < -eigvals_clip),
        lambda: backend.full_like(eigvals, backend.nan),
        lambda: eigvals * (eigvals >= eigvals_clip),
    )
