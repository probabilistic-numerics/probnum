"""Utility functions for linear operators."""
from __future__ import annotations

import numpy as np
import scipy.sparse

from probnum.typing import LinearOperatorLike

from . import _linear_operator


def aslinop(A: LinearOperatorLike) -> _linear_operator.LinearOperator:
    """Return ``A`` as a :class:`LinearOperator`.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable or object
        Argument to be represented as a linear operator. When `A` is an object it needs
        to have the attributes `.shape` and `.matvec`.

    Raises
    ------
    TypeError
        If :code:`A` can not be interpreted as a :class:`LinearOperator`.

    See Also
    --------
    LinearOperator : Class representing linear operators.

    Notes
    -----
    If `A` has no `.dtype` attribute, the data type is determined by calling
    :func:`LinearOperator.matvec()` - set the `.dtype` attribute to prevent this
    call upon the linear operator creation.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.linops import aslinop
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> aslinop(M)
    <Matrix with shape=(2, 3) and dtype=int32>
    """
    if isinstance(A, _linear_operator.LinearOperator):
        return A

    if isinstance(A, (np.ndarray, scipy.sparse.spmatrix)):
        return _linear_operator.Matrix(A=A)

    if isinstance(A, scipy.sparse.linalg.LinearOperator):
        return _linear_operator.LambdaLinearOperator(
            A.shape,
            A.dtype,
            matmul=_linear_operator.LinearOperator.broadcast_matmat(A.matmat),
        )

    raise TypeError(f"Cannot interpret {A} as a linear operator.")
