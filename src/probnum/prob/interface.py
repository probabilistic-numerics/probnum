"""
Convenience functions for random variables and distributions.
"""

import numpy as np
import scipy.sparse
import scipy.stats

from .distributions.distribution import Distribution
from .distributions.normal import Normal
from .distributions.dirac import Dirac


def asdist(obj):
    """
    Return ``obj`` as a :class:`Distribution`.

    Converts scalars, (sparse) arrays or distributions classes to a :class:`Distribution`.

    Parameters
    ----------
    obj : object
        Argument to be represented as a :class:`Distribution`.

    Returns
    -------
    dist : Distribution
        The object `obj` as a :class:`Distribution`.

    See Also
    --------
    Distribution : Class representing prob distributions.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> from probnum.prob import asdist
    >>> bern = bernoulli(p=0.5)
    >>> bern.random_state = 42  # Seed for reproducibility
    >>> d = asdist(bern)
    >>> d.sample(size=5)
    array([0, 1, 1, 1, 0])
    """
    # Distribution
    if isinstance(obj, Distribution):
        return obj
    # Scalar
    elif np.isscalar(obj):
        return Dirac(support=obj)
    # Sparse Array
    elif isinstance(obj, scipy.sparse.spmatrix):
        return Dirac(support=obj)
    # Linear Operator
    elif isinstance(obj, scipy.sparse.linalg.LinearOperator):
        return Dirac(support=obj)
    # Scipy distributions
    elif isinstance(obj, scipy.stats._distn_infrastructure.rv_frozen):
        return _scipystats_to_dist(obj=obj)
    else:
        try:
            # Numpy array
            return Dirac(support=np.array(obj))
        except Exception:
            raise NotImplementedError("Cannot convert object of type {} to a distributions.".format(type(obj)))


def _scipystats_to_dist(obj):
    """
    Transform scipy distributions to probnum distributions.

    Parameters
    ----------
    obj : object
        Scipy distribution.

    Returns
    -------
    dist : Distribution
        ProbNum distribution object.

    """
    # Normal distributions
    if obj.dist.name == "norm":
        return Normal(mean=obj.mean(), cov=obj.var(), random_state=obj.random_state)
    elif obj.__class__.__name__ == "multivariate_normal_frozen":  # Multivariate normal
        return Normal(mean=obj.mean, cov=obj.cov, random_state=obj.random_state)
    else:
        # Generic distributions
        if hasattr(obj, "pmf"):
            pdf = obj.pmf
            logpdf = obj.logpmf
        else:
            pdf = obj.pdf
            logpdf = obj.logpdf
        return Distribution(parameters={},
                            pdf=pdf,
                            logpdf=logpdf,
                            cdf=obj.cdf,
                            logcdf=obj.logcdf,
                            sample=obj.rvs,
                            mean=obj.mean,
                            var=obj.var,
                            random_state=obj.random_state)
