"""Normally distributed / Gaussian random variables."""

import functools
from typing import Optional, Union

import numpy as np
import scipy.linalg
import scipy.stats

from probnum import backend, config, linops, utils as _utils
from probnum.typing import (
    ArrayIndicesLike,
    ArrayLike,
    ArrayType,
    FloatLike,
    ScalarType,
    ShapeLike,
    ShapeType,
)

from . import _random_variable

_ValueType = Union[ArrayType, linops.LinearOperator]


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

        if not isinstance(cov_cholesky, linops.LinearOperator):
            cov = backend.asarray(cov)

        # Data type normalization
        dtype = backend.promote_types(mean.dtype, cov.dtype)

        if not backend.is_floating(dtype):
            dtype = backend.double

        mean = backend.cast(mean, dtype=dtype, casting="safe", copy=False)
        cov = backend.cast(cov, dtype=dtype, casting="safe", copy=False)

        if cov_cholesky is not None:
            # TODO: (#xyz) Handle if-statements like this via `pn.compat.cast`
            if isinstance(cov_cholesky, linops.LinearOperator):
                cov_cholesky = cov_cholesky.astype(dtype, casting="safe", copy=False)
            else:
                cov_cholesky = backend.cast(
                    cov_cholesky, dtype=dtype, casting="safe", copy=False
                )

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

        if cov_cholesky is not None:
            if cov_cholesky.shape != cov.shape:
                raise ValueError(
                    f"The cholesky decomposition of the covariance matrix must "
                    f"have the same shape as the covariance matrix, i.e. "
                    f"{cov.shape}, but shape {cov_cholesky.shape} was given"
                )

        self._cov_cholesky = cov_cholesky

        if mean.ndim == 0:
            # Scalar Gaussian
            if self._cov_cholesky is None:
                self._cov_cholesky = backend.sqrt(cov)

            self.__cov_op_cholesky = None

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
            self._cov_op = linops.aslinop(cov)
            self.__cov_op_cholesky = None

            if self._cov_cholesky is not None:
                self.__cov_op_cholesky = linops.aslinop(self._cov_cholesky)

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
    def dense_mean(self) -> ArrayType:
        """Dense representation of the mean."""
        if isinstance(self.mean, linops.LinearOperator):
            return self.mean.todense()

        return self.mean

    @property
    def dense_cov(self) -> ArrayType:
        """Dense representation of the covariance."""
        if isinstance(self.cov, linops.LinearOperator):
            return self.cov.todense()

        return self.cov

    # TODO (#xyz): Integrate Cholesky functionality into `LinearOperator.cholesky`

    @property
    def cov_cholesky(self) -> _ValueType:
        r"""Cholesky factor :math:`L` of the covariance
        :math:`\operatorname{Cov}(X) =LL^\top`."""

        if self._cov_cholesky is None:
            if isinstance(self.cov, linops.LinearOperator):
                self._cov_cholesky = self._cov_op_cholesky
            else:
                self._cov_cholesky = self._cov_matrix_cholesky

        return self._cov_cholesky

    @property
    def _cov_matrix_cholesky(self) -> ArrayType:
        return self._cov_op_cholesky.todense()

    @property
    def _cov_op_cholesky(self) -> _ValueType:
        if not self.cov_cholesky_is_precomputed:
            self.compute_cov_cholesky()

        return self.__cov_op_cholesky

    def compute_cov_cholesky(
        self,
        damping_factor: Optional[FloatLike] = None,
    ) -> None:
        """Compute Cholesky factor (careful: in-place operation!)."""
        if damping_factor is None:
            damping_factor = config.covariance_inversion_damping

        if self.cov_cholesky_is_precomputed:
            raise Exception("A Cholesky factor is already available.")

        if isinstance(self._cov_op, linops.Kronecker):
            A = self._cov_op.A.todense()
            B = self._cov_op.B.todense()

            self.__cov_op_cholesky = linops.Kronecker(
                A=backend.linalg.cholesky(
                    A + damping_factor * backend.eye(*A.shape, dtype=self.dtype),
                    lower=True,
                ),
                B=backend.linalg.cholesky(
                    B + damping_factor * backend.eye(*B.shape, dtype=self.dtype),
                    lower=True,
                ),
            )
        elif (
            isinstance(self._cov_op, linops.SymmetricKronecker)
            and self._cov_op.identical_factors
        ):
            A = self.cov.A.todense()

            self.__cov_op_cholesky = linops.SymmetricKronecker(
                A=backend.linalg.cholesky(
                    A + damping_factor * backend.eye(*A.shape, dtype=self.dtype),
                    lower=True,
                ),
            )
        else:
            self.__cov_op_cholesky = linops.aslinop(
                backend.linalg.cholesky(
                    self.dense_cov
                    + damping_factor * backend.eye(*self.shape, dtype=self.dtype),
                )
            )

    @property
    def cov_cholesky_is_precomputed(self) -> bool:
        """Return truth-value of whether the Cholesky factor of the covariance is
        readily available.

        This happens if (i) the Cholesky factor is specified during
        initialization or if (ii) the property `self.cov_cholesky` has
        been called before.
        """
        if self.__cov_op_cholesky is None:
            return False

        return True

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
    def _scalar_sample(
        self,
        rng: np.random.Generator,
        size: ShapeType = (),
    ) -> ArrayType:
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
    @backend.jit
    def _scalar_in_support(x: _ValueType) -> ArrayType:
        return backend.isfinite(x)

    @backend.jit_method
    def _scalar_pdf(self, x: _ValueType) -> ArrayType:
        return backend.exp(-((x - self.mean) ** 2) / (2.0 * self.var)) / backend.sqrt(
            2 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_logpdf(self, x: _ValueType) -> ArrayType:
        return -((x - self.mean) ** 2) / (2.0 * self.var) - 0.5 * backend.log(
            2.0 * backend.pi * self.var
        )

    @backend.jit_method
    def _scalar_cdf(self, x: _ValueType) -> ArrayType:
        return backend.special.ndtr((x - self.mean) / self.std)

    @backend.jit_method
    def _scalar_logcdf(self, x: _ValueType) -> ArrayType:
        return backend.log(self._scalar_cdf(x))

    @backend.jit_method
    def _scalar_quantile(self, p: FloatLike) -> ArrayType:
        return self.mean + self.std * backend.special.ndtri(p)

    @backend.jit_method
    def _scalar_entropy(self) -> ScalarType:
        return 0.5 * backend.log(2.0 * backend.pi * self.var) + 0.5

    # Multi- and matrixvariate Gaussians
    def _sample(self, rng: np.random.Generator, size: ShapeType = ()) -> _ValueType:
        sample = scipy.stats.multivariate_normal.rvs(
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
            size=size,
            random_state=rng,
        )

        return sample.reshape(sample.shape[:-1] + self.shape)

    @staticmethod
    def _arg_todense(x: Union[ArrayType, linops.LinearOperator]) -> ArrayType:
        if isinstance(x, linops.LinearOperator):
            return x.todense()
        elif isinstance(x, backend.ndarray):
            return x
        else:
            raise ValueError(f"Unsupported argument type {type(x)}")

    @backend.jit_method
    def _in_support(self, x: _ValueType) -> ArrayType:
        return backend.all(
            backend.isfinite(Normal._arg_todense(x)),
            axis=tuple(range(-self.ndim, 0)),
            keepdims=False,
        )

    @backend.jit_method
    def _pdf(self, x: _ValueType) -> ArrayType:
        return backend.exp(self._logpdf(x))

    @backend.jit_method
    def _logpdf(self, x: _ValueType) -> ArrayType:
        x_centered = Normal._arg_todense(x - self.dense_mean).reshape(
            x.shape[: -self.ndim] + (-1,)
        )[..., None]

        res = (
            -0.5
            * (
                x_centered.T
                # TODO (#xyz): Replace `cho_solve` with linop.cholesky().solve()
                @ backend.linalg.cholesky_solve(
                    (self._cov_matrix_cholesky, True), x_centered
                )
            )[..., 0, 0]
        )

        res -= 0.5 * self.size * backend.log(backend.array(2.0 * backend.pi))
        # TODO (#xyz): Replace this with `0.5 * self._cov_op.logdet()`
        res -= backend.sum(backend.log(backend.diag(self._cov_matrix_cholesky)))

        return res

    def _cdf(self, x: _ValueType) -> ArrayType:
        if backend.BACKEND is not backend.Backend.NUMPY:
            raise NotImplementedError()

        return scipy.stats.multivariate_normal.cdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    def _logcdf(self, x: _ValueType) -> ArrayType:
        if backend.BACKEND is not backend.Backend.NUMPY:
            raise NotImplementedError()

        return scipy.stats.multivariate_normal.logcdf(
            Normal._arg_todense(x).reshape(x.shape[: -self.ndim] + (-1,)),
            mean=self.dense_mean.ravel(),
            cov=self.dense_cov,
        )

    @backend.jit_method
    def _var(self) -> ArrayType:
        return backend.diag(self.dense_cov).reshape(self.shape)

    @backend.jit_method
    def _entropy(self) -> ScalarType:
        entropy = 0.5 * self.size * (backend.log(2.0 * backend.pi) + 1.0)
        # TODO (#xyz): Replace this with `0.5 * self._cov_op.logdet()`
        entropy += backend.sum(backend.log(backend.diag(self._cov_matrix_cholesky)))

        return entropy
