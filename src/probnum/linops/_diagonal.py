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

        if isinstance(diagonal, np.ndarray) and diagonal.ndim == 1:
            # Anisotropic scaling
            self._diagonal = diagonal.astype(dtype, copy=False)

            shape = (self._diagonal.shape[0], self._diagonal.shape[0])
            dtype = self._diagonal.dtype

            apply = lambda x, axis: (
                self._diagonal.reshape((-1,) + (x.ndim - 1 - axis) * (1,)) * x
            )

            matmul = lambda x: self._diagonal[:, np.newaxis] * x
            rmatmul = lambda x: self._diagonal * x

            todense = lambda: np.diag(self._diagonal)

            if np.issubdtype(dtype, np.complexfloating):
                adjoint = lambda: Diagonal(np.conj(self._diagonal))
            else:
                adjoint = lambda: self

            inverse = self._inverse_anisotropic

            rank = lambda: np.count_nonzero(self.diagonal, axis=0)
            eigvals = lambda: self._diagonal
            cond = self._cond_anisotropic
            det = lambda: np.prod(self._diagonal).astype(self._inexact_dtype)
            logabsdet = None
            trace = lambda: np.sum(self._diagonal)
        elif np.ndim(diagonal) == 0:
            self._scalar = probnum.utils.as_numpy_scalar(diagonal, dtype=dtype)

            if shape is None:
                raise ValueError(
                    "When specifying the diagonal entries by a scalar, a shape must be"
                    "specified."
                )

            shape = probnum.utils.as_shape(shape)

            if len(shape) == 1:
                shape = shape + shape
            elif len(shape) > 2:
                raise ValueError(
                    "The shape of a linear operator must be two-dimensional."
                )

            dtype = self._scalar.dtype

            if self._scalar == 1:
                # Identity
                apply = lambda x, axis: x

                matmul = lambda x: x
                rmatmul = lambda x: x

                todense = lambda: np.identity(shape[0], dtype=dtype)

                adjoint = lambda: self
                inverse = lambda: self

                rank = lambda: np.intp(shape[0])
                cond = self._cond_isotropic
                eigvals = lambda: np.ones(shape[0], dtype=self._inexact_dtype)
                det = lambda: probnum.utils.as_numpy_scalar(
                    1.0, dtype=self._inexact_dtype
                )
                logabsdet = lambda: probnum.utils.as_numpy_scalar(
                    0.0, dtype=self._inexact_dtype
                )
                trace = lambda: probnum.utils.as_numpy_scalar(
                    self.shape[0], dtype=self.dtype
                )
            else:
                # Isotropic scaling
                apply = lambda x, axis: self._scalar * x

                matmul = lambda x: self._scalar * x
                rmatmul = lambda x: self._scalar * x

                todense = lambda: np.diag(np.full(shape[0], self._scalar, dtype=dtype))

                if np.imag(self._scalar) != 0.0:
                    adjoint = lambda: Diagonal(np.conj(self._scalar), shape=shape)
                else:
                    adjoint = lambda: self

                inverse = self._inv_isotropic

                if self._scalar == 0:
                    rank = lambda: np.intp(0)
                    cond = lambda p: probnum.utils.as_numpy_scalar(
                        np.inf, dtype=self._inexact_dtype
                    )
                else:
                    rank = lambda: np.intp(shape[0])
                    cond = self._cond_isotropic

                eigvals = lambda: np.full(
                    shape[0], self._scalar, dtype=self._inexact_dtype
                )
                det = lambda: self._scalar.astype(self._inexact_dtype) ** shape[0]
                logabsdet = lambda: shape[0] * np.log(np.abs(self._scalar))
                trace = lambda: self.shape[0] * self._scalar

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

        if not self.is_square:
            raise np.linalg.LinAlgError("Diagonal linear operators must be square.")

    @property
    def diagonal(self):
        if self._diagonal is None:
            self._diagonal = np.full(self.shape[0], self._scalar, dtype=self._dtype)

        return self._diagonal

    def _inverse_anisotropic(self) -> "Diagonal":
        if self.rank() != self.shape[0]:
            raise np.linalg.LinAlgError("The operator is singular.")

        return Diagonal(1 / self._diagonal, shape=self.shape)

    def _inv_isotropic(self) -> "Diagonal":
        if self._scalar == 0:
            raise np.linalg.LinAlgError("The operator is singular")

        return Diagonal(1 / self._scalar, shape=self.shape)

    def _cond_anisotropic(self, p) -> np.number:
        if p is None or p == 1 or p == 2 or p == np.inf:
            abs_diag = np.abs(self._diagonal)
            return np.max(abs_diag) / np.min(abs_diag)

        return np.linalg.cond(self.todense(), p=p)

    def _cond_isotropic(self, p) -> np.inexact:
        if p is None or p in (2, 1, np.inf, -2, -1, -np.inf):
            return probnum.utils.as_numpy_scalar(1.0, dtype=self._inexact_dtype)
        elif p == "fro":
            return probnum.utils.as_numpy_scalar(
                min(self.shape), dtype=self._inexact_dtype
            )
        else:
            return np.linalg.cond(self.todense(), p=p)


class ScalarMult(Diagonal):
    """A linear operator representing scalar multiplication.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (N, N).
    scalar : float
        Scalar to multiply by.
    """

    def __init__(
        self,
        shape: ShapeArgType,
        scalar: ScalarArgType,
        dtype: Optional[DTypeArgType] = None,
    ):
        super().__init__(diagonal=scalar, shape=shape, dtype=dtype)

    @property
    def scalar(self):
        return self._scalar


class Identity(ScalarMult):
    """The identity operator.

    Parameters
    ----------
    shape : int or tuple
        Shape of the identity operator.
    """

    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType = np.float_,
    ):
        super().__init__(shape, 1, dtype)
