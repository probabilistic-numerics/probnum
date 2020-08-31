import numpy as np
import scipy.sparse

from . import _random_variable
from . import _dirac
from . import _scipy_stats


def asrandvar(obj) -> _random_variable.RandomVariable:
    """
    Return ``obj`` as a :class:`RandomVariable`.

    Converts scalars, (sparse) arrays or distribution classes to a
    :class:`RandomVariable`.

    Parameters
    ----------
    obj : object
        Argument to be represented as a :class:`RandomVariable`.

    Returns
    -------
    rv : RandomVariable
        The object `obj` as a :class:`RandomVariable`.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> from probnum import asrandvar
    >>> bern = bernoulli(p=0.5)
    >>> bern.random_state = 42  # Seed for reproducibility
    >>> b = asrandvar(bern)
    >>> b.sample(size=5)
    array([1, 1, 1, 0, 0])
    """

    # RandomVariable
    if isinstance(obj, _random_variable.RandomVariable):
        return obj
    # Scalar
    elif np.isscalar(obj):
        return _dirac.Dirac(support=obj)
    # Numpy array, sparse array or Linear Operator
    elif isinstance(
        obj, (np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator)
    ):
        return _dirac.Dirac(support=obj)
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
