"""Utility functions for random variables."""
from typing import Any

import numpy as np
import scipy.sparse

from . import _constant, _random_variable, _scipy_stats


def asrandvar(obj: Any) -> _random_variable.RandomVariable:
    """Convert ``obj`` to a :class:`RandomVariable`.

    Converts an object such as scalars, (sparse) arrays, or distribution-type objects to
    a ProbNum :class:`RandomVariable`.

    Parameters
    ----------
    obj :
        Object to be represented as a :class:`RandomVariable`.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> import probnum as pn
    >>> bern = bernoulli(p=0.5)
    >>> bern.random_state = 42  # Seed for reproducibility
    >>> bern_pn = pn.asrandvar(bern)
    >>> bern_pn.sample(size=5)
    array([1, 1, 1, 0, 0])
    """

    # pylint: disable=protected-access

    # RandomVariable
    if isinstance(obj, _random_variable.RandomVariable):
        return obj
    # Scalar
    elif np.isscalar(obj):
        return _constant.Constant(support=obj)
    # Numpy array, sparse array or linear operator
    elif isinstance(
        obj, (np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator)
    ):
        return _constant.Constant(support=obj)
    # Scipy random variable
    elif isinstance(
        obj,
        (
            scipy.stats._distn_infrastructure.rv_frozen,
            scipy.stats._multivariate.multi_rv_frozen,
        ),
    ):
        return _scipy_stats.wrap_scipy_rv(obj)
    else:
        raise ValueError(
            f"Argument of type {type(obj)} cannot be converted to a random variable."
        )
