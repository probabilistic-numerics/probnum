"""Vectorization utilities for matrix-{vector, matrix} products."""

import functools
from typing import Callable, Optional, Union

import numpy as np

_VectorizationDecoratorReturnType = Union[
    Callable[[np.ndarray], np.ndarray],
    Callable[[Callable[[np.ndarray], np.ndarray]], Callable[[np.ndarray], np.ndarray]],
]


def vectorize_matmat(
    matmat: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    method: bool = False,
) -> _VectorizationDecoratorReturnType:
    """Broadcasting for a (implicitly defined) matrix-matrix product.

    Convenience function / decorator to broadcast the definition of a matrix-matrix
    product to stacks of matrices. This can be used to easily construct a new linear
    operator only from a matrix-matrix product.

    Parameters
    ----------
    matmat
        Function computing a matrix-matrix product.
    method
        Whether the decorator is being applied to a method or a function.
    """

    def _vectorize_matmat(
        matmat: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        np_vectorize_obj = np.vectorize(
            matmat,
            excluded={0} if method else None,
            signature="(n,k)->(m,k)",
        )

        # The additional wrapper function is needed when using this as a method
        # decorator, since the class np.vectorize is not implemented as a descriptor
        @functools.wraps(matmat)
        def vectorized_matmat(*args) -> np.ndarray:
            return np_vectorize_obj(*args)

        return vectorized_matmat

    if matmat is None:
        return _vectorize_matmat

    return _vectorize_matmat(matmat)


def vectorize_matvec(
    matvec: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    method: bool = False,
) -> _VectorizationDecoratorReturnType:
    """Broadcasting for a (implicitly defined) matrix-vector product.

    Convenience function / decorator to broadcast the definition of a matrix-vector
    product. This can be used to easily construct a new linear operator only from a
    matrix-vector product.

    Parameters
    ----------
    matvec
        Function computing a matrix-vector product.
    method
        Whether the decorator is being applied to a method or a function.
    """

    def _vectorize_matvec(
        matvec: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        @functools.wraps(matvec)
        def vectorized_matvec(*args) -> np.ndarray:
            x = args[1 if method else 0]

            if x.ndim == 2 and x.shape[1] == 1:
                return matvec(x[:, 0])[:, np.newaxis]

            return np.apply_along_axis(matvec, -2, x)

        return vectorized_matvec

    if matvec is None:
        return _vectorize_matvec

    return _vectorize_matvec(matvec)
