"""LinearOperator that represents pairwise covariances of evaluations."""

from typing import Callable, Optional
import warnings

import numpy as np

from probnum import linops
from probnum.typing import ShapeType

_USE_KEOPS = True
try:
    from pykeops.numpy import LazyTensor
except ImportError:  # pragma: no cover
    _USE_KEOPS = False
    warnings.warn(
        "KeOps is not installed and currently unavailable for Windows."
        "This may prevent scaling to large datasets."
    )


class CovarianceLinearOperator(linops.LinearOperator):
    """:class:`~probnum.linops.LinearOperator` representing the pairwise
    covariances of evaluations of :math:`f_0` and :math:`f_1` at the given input
    points.

    Supports both KeOps-based and standard implementations, but will prefer KeOps-based
    implementations by default.

    Parameters
        ----------
        x0
            *shape=* ``(prod(batch_shape_0),) +`` :attr:`input_shape_0` -- (Batch of)
            input(s) for the first argument of the :class:`CovarianceFunction`.
        x1
            *shape=* ``(prod(batch_shape_1),) +`` :attr:`input_shape_1` -- (Batch of)
            input(s) for the second argument of the :class:`CovarianceFunction`.
            Can also be set to :data:`None`, in which case the function will behave as
            if ``x1 == x0`` (potentially using a more efficient implementation for this
            particular case).
        shape
            Shape of the linear operator.
        evaluate_dense_matrix
            Callable for the standard implementation that evaluates k(x0, x1) densely.
        keops_lazy_tensor
            :class:`~pykeops.numpy.LazyTensor` representing the covariance matrix
            corresponding to the given batches of input points.
    """

    def __init__(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
        shape: ShapeType,
        evaluate_dense_matrix: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        keops_lazy_tensor: Optional["LazyTensor"] = None,
    ):
        self._x0 = x0
        self._x1 = x1

        self._evaluate_dense_matrix = evaluate_dense_matrix
        self._keops_lazy_tensor = keops_lazy_tensor
        self._use_keops = _USE_KEOPS and self._keops_lazy_tensor is not None
        dtype = np.promote_types(x0.dtype, x1.dtype) if x1 is not None else x0.dtype
        super().__init__(shape, dtype)

    @property
    def keops_lazy_tensor(self) -> Optional["LazyTensor"]:
        """:class:`~pykeops.numpy.LazyTensor` representing the covariance matrix
        corresponding to the given batches of input points.
        When not using KeOps, this is set to :data:`None`.
        """
        return self._keops_lazy_tensor

    def _todense(self) -> np.ndarray:
        return self._evaluate_dense_matrix(self._x0, self._x1)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._use_keops:
            return self.keops_lazy_tensor @ x
        return self.todense() @ x

    def _transpose(self) -> linops.LinearOperator:
        return CovarianceLinearOperator(
            self._x0,
            self._x1,
            (self.shape[1], self.shape[0]),
            lambda x0, x1: self._evaluate_dense_matrix(x0, x1).T,
            self._keops_lazy_tensor.T if self._keops_lazy_tensor is not None else None,
        )
