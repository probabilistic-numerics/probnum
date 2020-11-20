import numpy as np
import scipy.sparse

from . import _linear_operator


def aslinop(A) -> _linear_operator.LinearOperator:
    """Return ``A`` as a :class:`LinearOperator`.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable or object
        Argument to be represented as a linear operator. When `A` is an object it needs
        to have the attributes `.shape` and `.matvec`.

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
    <2x3 MatrixMult with dtype=int32>
    """
    if isinstance(A, scipy.sparse.linalg.LinearOperator):
        return A
    elif isinstance(A, (np.ndarray, scipy.sparse.spmatrix)):
        return _linear_operator.MatrixMult(A=A)
    else:
        op = scipy.sparse.linalg.aslinearoperator(A)
        return _linear_operator.LinearOperator(op)
