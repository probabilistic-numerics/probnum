from typing import Optional, Union

import numpy as np

import probnum.utils
from probnum.type import DTypeArgType, ScalarArgType, ShapeArgType

from . import _linear_operator


class Diagonal(_linear_operator.LinearOperator):
    def __init__(
        self,
        diagonal: Union[np.ndarray, ScalarArgType],
        shape: Optional[ShapeArgType] = None,
        dtype: Optional[DTypeArgType] = None,
    ):
        self._diagonal = None
        self._scalar = None

        if np.ndim(diagonal) == 0:
            # Isotropic scaling
            self._scalar = probnum.utils.as_numpy_scalar(diagonal, dtype=dtype)

            if shape is None:
                raise ValueError(
                    "When specifying the diagonal entries by a scalar, a shape must be"
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
                raise np.linalg.LinAlgError("Diagonal linear operators must be square.")

            dtype = self._scalar.dtype

            if self._scalar == 1:
                # Identity
                matmul = lambda x: x.astype(
                    np.result_type(self.dtype, x.dtype), copy=False
                )
                rmatmul = lambda x: x.astype(
                    np.result_type(self.dtype, x.dtype), copy=False
                )

                apply = lambda x, axis: x.astype(
                    np.result_type(self.dtype, x.dtype), copy=False
                )

                todense = lambda: np.identity(shape[0], dtype=dtype)

                adjoint = lambda: self
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
                rmatmul = lambda x: self._scalar * x

                apply = lambda x, axis: self._scalar * x

                todense = self._todense_isotropic

                adjoint = lambda: (
                    self
                    if np.imag(self._scalar) == 0
                    else Diagonal(np.conj(self._scalar), shape=shape)
                )
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
        elif np.ndim(diagonal) == 1:
            # Anisotropic scaling
            self._diagonal = np.asarray(diagonal, dtype=dtype)
            self._diagonal.setflags(write=False)

            shape = 2 * self._diagonal.shape
            dtype = self._diagonal.dtype

            matmul = lambda x: self._diagonal[:, np.newaxis] * x
            rmatmul = lambda x: self._diagonal * x

            apply = lambda x, axis: (
                self._diagonal.reshape((-1,) + (x.ndim - (axis + 1)) * (1,)) * x
            )

            todense = lambda: np.diag(self._diagonal)

            adjoint = lambda: (
                self
                if (
                    not np.issubdtype(dtype, np.complexfloating)
                    or np.all(np.imag(self._diagonal) == 0)
                )
                else Diagonal(np.conj(self._diagonal))
            )
            inverse = self._inverse_anisotropic

            rank = lambda: np.count_nonzero(self.diagonal, axis=0)
            eigvals = lambda: self._diagonal
            cond = self._cond_anisotropic
            det = lambda: np.prod(self._diagonal).astype(
                self._inexact_dtype, copy=False
            )
            logabsdet = None
            trace = lambda: np.sum(self._diagonal)
        else:
            raise TypeError(
                "`diagonal` must either be a scalar or a 1-dimensional array-like"
            )

        super().__init__(
            shape,
            dtype,
            matmul=matmul,
            rmatmul=rmatmul,
            apply=apply,
            todense=todense,
            transpose=lambda: self,
            adjoint=adjoint,
            inverse=inverse,
            rank=rank,
            eigvals=eigvals,
            cond=cond,
            det=det,
            logabsdet=logabsdet,
            trace=trace,
        )

    @property
    def diagonal(self) -> np.ndarray:
        if self._diagonal is None:
            self._diagonal = np.full(self.shape[0], self._scalar, dtype=self._dtype)

        return self._diagonal

    @property
    def scalar(self) -> Optional[np.number]:
        return self._scalar

    @property
    def is_isotropic(self) -> bool:
        return self._scalar is not None

    def _astype(self, dtype, order, casting, copy) -> "Diagonal":
        if self.dtype == dtype and not copy:
            return self
        else:
            if self.is_isotropic:
                return Diagonal(self._scalar, shape=self.shape, dtype=dtype)
            else:
                return Diagonal(self._diagonal, dtype=dtype)

    def _todense_isotropic(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=self.dtype)
        np.fill_diagonal(dense, self._scalar)
        return dense

    def _inverse_anisotropic(self) -> "Diagonal":
        if self.rank() < self.shape[0]:
            raise np.linalg.LinAlgError("The operator is singular.")

        return Diagonal(1 / self._diagonal, shape=self.shape)

    def _inverse_isotropic(self) -> "Diagonal":
        if self.rank() < self.shape[0]:
            raise np.linalg.LinAlgError("The operator is singular.")

        return Diagonal(1 / self._scalar, shape=self.shape)

    def _cond_anisotropic(self, p) -> np.inexact:
        abs_diag = np.abs(self._diagonal)
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
                if abs_max > 0:
                    cond = abs_min / abs_max
                else:
                    cond = np.double(np.inf)

            return cond.astype(self._inexact_dtype, copy=False)
        elif p == "fro":
            norm = np.linalg.norm(self._diagonal, ord=2)
            norm_inv = np.linalg.norm(1 / self._diagonal, ord=2)
            return (norm * norm_inv).astype(self._inexact_dtype, copy=False)

        return np.linalg.cond(self.todense(cache=False), p=p)

    def _cond_isotropic(self, p) -> np.inexact:
        if self._scalar == 0:
            return self._inexact_dtype.type(np.inf)

        if p is None or p in (2, 1, np.inf, -2, -1, -np.inf):
            return probnum.utils.as_numpy_scalar(1.0, dtype=self._inexact_dtype)
        elif p == "fro":
            return probnum.utils.as_numpy_scalar(
                min(self.shape), dtype=self._inexact_dtype
            )
        else:
            return np.linalg.cond(self.todense(cache=False), p=p)


class Identity(Diagonal):
    """The identity operator.

    Parameters
    ----------
    shape :
        Shape of the identity operator.
    """

    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType = np.double,
    ):
        super().__init__(1, shape=shape, dtype=dtype)

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "Identity":
        if dtype == self.dtype and not copy:
            return self
        else:
            return Identity(self.shape, dtype=dtype)
