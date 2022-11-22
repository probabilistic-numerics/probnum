"""Utility functions for random variables."""
from typing import Any

import scipy.sparse

from probnum import backend, linops

from . import _constant, _random_variable


def asrandvar(obj: Any) -> _random_variable.RandomVariable:
    """Convert ``obj`` to a :class:`RandomVariable`.

    Converts an object such as scalars, (sparse) arrays, or distribution-type objects to
    a ProbNum :class:`RandomVariable`.

    Parameters
    ----------
    obj
        Object to be represented as a :class:`RandomVariable`.

    Returns
    -------
    randvar
        Object as a :class:`RandomVariable`.

    Raises
    ------
    ValueError
        If the object cannot be represented as a :class:`RandomVariable`.

    See Also
    --------
    RandomVariable : Class representing random variables.
    """

    # RandomVariable
    if isinstance(obj, _random_variable.RandomVariable):
        return obj

    # Scalar
    if backend.ndim(obj) == 0:
        return _constant.Constant(support=obj)

    # NumPy array or sparse matrix
    if backend.isarray(obj) or isinstance(obj, scipy.sparse.spmatrix):
        return _constant.Constant(support=obj)

    # Linear Operators
    if isinstance(obj, (linops.LinearOperator, scipy.sparse.linalg.LinearOperator)):
        return _constant.Constant(support=linops.aslinop(obj))

    raise ValueError(
        f"Argument of type {type(obj)} cannot be converted to a random variable."
    )
