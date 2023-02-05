"""Scaling linear operator."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from probnum.typing import DTypeLike, ScalarLike, ShapeLike
import probnum.utils

from . import _linear_operator

# pylint: disable="too-many-statements"


class Scaling(_linear_operator.LambdaLinearOperator):
    r"""Scaling linear operator.

    Creates a diagonal linear operator which (non-uniformly)
    scales elements of vectors, defined by

    .. math::
        v \mapsto \begin{bmatrix}
            \alpha_1 & 0 & \dots   & 0 \\
            0   &  \alpha_2 &   & \vdots \\
            \vdots  &   &  \ddots  & 0 \\
            0 &  \dots  & 0 & \alpha_n
        \end{bmatrix} v.

    Parameters
    ----------
    factors:
        Scaling factor(s) on the diagonal.
    shape :
        Shape of the linear operator.
    dtype :
        Data type of the linear operator.

    """

    def __init__(
        self,
        factors: Union[np.ndarray, ScalarLike],
        shape: Optional[ShapeLike] = None,
        dtype: Optional[DTypeLike] = None,
    ):
        self._factors = None
        self._scalar = None

        if np.ndim(factors) == 0:
            # Isotropic scaling
            self._scalar = probnum.utils.as_numpy_scalar(factors, dtype=dtype)

            if shape is None:
                raise ValueError(
                    "When specifying the scaling factors by a scalar, a shape must be "
                    "specified."
                )

            shape = probnum.utils.as_shape(shape)

            if len(shape) == 1:
                shape = 2 * shape
            elif len(shape) != 2:
                raise ValueError(
                    "The shape of a linear operator must be two-dimensional."
                )

            if shape[0] != shape[1]:
                raise np.linalg.LinAlgError("Scaling operators must be square.")

            dtype = self._scalar.dtype

            if self._scalar == 1:
                # Identity
                matmul = lambda x: x.astype(
                    np.result_type(self.dtype, x.dtype), copy=False
                )

                apply = lambda x, axis: x.astype(
                    np.result_type(self.dtype, x.dtype), copy=False
                )

                todense = lambda: np.identity(shape[0], dtype=dtype)

                inverse = lambda: self

                rank = lambda: np.intp(shape[0])
                cond = self._cond_isotropic
                eigvals = lambda: np.ones(shape[0], dtype=self._inexact_dtype)
                det = lambda: self._scalar.astype(self._inexact_dtype, copy=False)
                logabsdet = lambda: (0 * self._scalar).astype(
                    self._inexact_dtype, copy=False
                )
            else:
                matmul = lambda x: self._scalar * x

                apply = lambda x, axis: self._scalar * x

                todense = self._todense_isotropic

                inverse = self._inverse_isotropic

                rank = lambda: np.intp(0 if self._scalar == 0 else shape[0])
                cond = self._cond_isotropic
                eigvals = lambda: np.full(
                    shape[0], self._scalar, dtype=self._inexact_dtype
                )
                det = lambda: (
                    self._scalar.astype(self._inexact_dtype, copy=False) ** shape[0]
                )
                logabsdet = lambda: (
                    probnum.utils.as_numpy_scalar(-np.inf, dtype=self._inexact_dtype)
                    if self._scalar == 0
                    else shape[0] * np.log(np.abs(self._scalar))
                )

            trace = lambda: self.shape[0] * self._scalar
        elif np.ndim(factors) == 1:
            # Anisotropic scaling
            self._factors = np.asarray(factors, dtype=dtype)
            self._factors.setflags(write=False)

            shape = 2 * self._factors.shape
            dtype = self._factors.dtype

            matmul = lambda x: self._factors[:, None] * x

            apply = lambda x, axis: (
                self._factors.reshape((-1,) + (x.ndim - (axis + 1)) * (1,)) * x
            )

            todense = lambda: np.diag(self._factors)

            inverse = self._inverse_anisotropic

            rank = lambda: np.count_nonzero(self.factors, axis=0)
            eigvals = lambda: self._factors
            cond = self._cond_anisotropic
            det = lambda: np.prod(self._factors).astype(self._inexact_dtype, copy=False)
            logabsdet = None
            trace = lambda: np.sum(self._factors)
        else:
            raise TypeError(
                "`factors` must either be a scalar or a 1-dimensional array-like"
            )

        super().__init__(
            shape,
            dtype,
            matmul=matmul,
            apply=apply,
            solve=lambda B: self.inv() @ B,
            todense=todense,
            transpose=lambda: self,
            inverse=inverse,
            rank=rank,
            eigvals=eigvals,
            cond=cond,
            det=det,
            logabsdet=logabsdet,
            trace=trace,
        )

        # Matrix properties
        self.is_symmetric = True
        self.is_lower_triangular = True
        self.is_upper_triangular = True

        if self._scalar is not None:
            self.is_positive_definite = bool(self._scalar > 0.0)

    @property
    def factors(self) -> np.ndarray:
        """Scaling factors.

        Scaling factors on the diagonal of the matrix representation.
        """
        if self._factors is None:
            self._factors = np.full(self.shape[0], self._scalar, dtype=self.dtype)

        return self._factors

    @property
    def scalar(self) -> Optional[np.number]:
        """Scaling factor."""
        return self._scalar

    @property
    def is_isotropic(self) -> bool:
        """Whether scaling is uniform / isotropic."""
        if self._scalar is None and np.all(self.factors == self.factors[0]):
            self._scalar = self.factors[0]
        return self._scalar is not None

    def __eq__(self, other: _linear_operator.LinearOperator) -> bool:
        if not self._is_type_shape_dtype_equal(other):
            return False

        if self.is_isotropic and other.is_isotropic:
            return self.scalar == other.scalar

        if not self.is_isotropic and not self.is_isotropic:
            return np.all(self.factors == other.factors)

        return False

    def _add_scaling(self, other: "Scaling") -> "Scaling":
        if other.shape != self.shape:
            raise ValueError(
                "Addition of two Scaling LinearOperators is only "
                "possible if both operands have the same shape."
            )

        return Scaling(
            factors=self.factors + other.factors, shape=self.shape, dtype=self.dtype
        )

    def _sub_scaling(self, other: "Scaling") -> "Scaling":
        if other.shape != self.shape:
            raise ValueError(
                "Subtraction of two Scaling LinearOperators is only "
                "possible if both operands have the same shape."
            )

        return Scaling(
            factors=self.factors - other.factors, shape=self.shape, dtype=self.dtype
        )

    def _mul_scaling(self, other: "Scaling") -> "Scaling":
        if other.shape != self.shape:
            raise ValueError(
                "Multiplication of two Scaling LinearOperators is only "
                "possible if both operands have the same shape."
            )

        return Scaling(
            factors=self.factors * other.factors, shape=self.shape, dtype=self.dtype
        )

    def _matmul_scaling(self, other: "Scaling") -> "Scaling":
        return self._mul_scaling(other)

    def _astype(self, dtype, order, casting, copy) -> "Scaling":
        if self.dtype == dtype and not copy:
            return self

        if self.is_isotropic:
            return Scaling(self._scalar, shape=self.shape, dtype=dtype)

        return Scaling(self._factors, dtype=dtype)

    def _todense_isotropic(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=self.dtype)
        np.fill_diagonal(dense, self._scalar)
        return dense

    def _inverse_anisotropic(self) -> "Scaling":
        if self.rank() < self.shape[0]:
            raise np.linalg.LinAlgError("The operator is singular.")

        return Scaling(1 / self._factors)

    def _inverse_isotropic(self) -> "Scaling":
        if self.rank() < self.shape[0]:
            raise np.linalg.LinAlgError("The operator is singular.")

        return Scaling(1 / self._scalar, shape=self.shape)

    def _cond_anisotropic(self, p: Union[None, int, float, str]) -> np.floating:
        abs_diag = np.abs(self._factors)
        abs_min = np.min(abs_diag)

        if abs_min == 0.0:
            # The operator is singular
            return probnum.utils.as_numpy_scalar(np.inf, dtype=self._inexact_dtype)

        if p is None:
            p = 2

        if p in (2, 1, np.inf, -2, -1, -np.inf):
            abs_max = np.max(abs_diag)

            if p > 0:  # p in (2, 1, np.inf)
                cond = abs_max / abs_min
            else:  # p in (-2, -1, -np.inf)
                if abs_max > 0:  # pylint: disable=else-if-used
                    cond = abs_min / abs_max
                else:
                    cond = np.double(np.inf)

            return cond.astype(self._inexact_dtype, copy=False)

        if p == "fro":
            norm = np.linalg.norm(self._factors, ord=2)
            norm_inv = np.linalg.norm(1 / self._factors, ord=2)
            return (norm * norm_inv).astype(self._inexact_dtype, copy=False)

        return np.linalg.cond(self.todense(cache=False), p=p)

    def _cond_isotropic(self, p: Union[None, int, float, str]) -> np.floating:
        if self._scalar == 0:
            return self._inexact_dtype.type(np.inf)

        if p is None or p in (2, 1, np.inf, -2, -1, -np.inf):
            return probnum.utils.as_numpy_scalar(1.0, dtype=self._inexact_dtype)

        if p == "fro":
            return probnum.utils.as_numpy_scalar(
                min(self.shape), dtype=self._inexact_dtype
            )

        return np.linalg.cond(self.todense(cache=False), p=p)

    def _cholesky(self, lower: bool = True) -> Scaling:
        if self._scalar is not None:
            if self._scalar <= 0:
                raise np.linalg.LinAlgError(
                    "The linear operator is not positive definite."
                )

            return Scaling(np.sqrt(self._scalar), shape=self.shape)

        if np.any(self._factors <= 0):
            raise np.linalg.LinAlgError("The linear operator is not positive definite.")

        return Scaling(np.sqrt(self._factors))


class Zero(_linear_operator.LambdaLinearOperator):
    """The zero matrix represented as a :class:`LinearOperator`.

    Linear operator that maps all inputs to 0.

    Parameters
    ----------
    shape :
        Shape of the linear operator.
    dtype :
        Data type of the linear operator.
    """

    def __init__(self, shape, dtype=np.float64):
        todense = lambda: np.zeros(shape=shape, dtype=dtype)
        rank = lambda: np.intp(0)
        eigvals = lambda: np.zeros(shape=(shape[0],), dtype=dtype)
        det = lambda: np.zeros(shape=(), dtype=dtype)

        trace = lambda: np.zeros(shape=(), dtype=dtype)

        def matmul(x: np.ndarray) -> np.ndarray:
            target_shape = list(x.shape)
            target_shape[-2] = self.shape[0]
            return np.zeros(target_shape, np.result_type(x, self.dtype))

        super().__init__(
            shape,
            dtype=dtype,
            matmul=matmul,
            todense=todense,
            transpose=lambda: Zero(self.shape[::-1], self.dtype),
            rank=rank,
            eigvals=eigvals,
            det=det,
            trace=trace,
        )

        # Matrix properties
        if self.is_square:
            self.is_symmetric = True
            self.is_lower_triangular = True
            self.is_upper_triangular = True

            self.is_positive_definite = False
