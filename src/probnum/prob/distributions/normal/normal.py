"""
Normal distribution.

This module implements the Gaussian distribution in its
univariate, multi-variate, matrix-variate and
operator-variate forms.
"""

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.linalg import linops
from probnum.prob.distributions.normal._univariatenormal import _UnivariateNormal
from probnum.prob.distributions.normal._multivariatenormal import _MultivariateNormal, _MatrixvariateNormal
from probnum.prob.distributions.normal._operatorvariatenormal import _OperatorvariateNormal, _SymmetricKroneckerIdenticalFactorsNormal

__all__ = ["Normal"]

class Normal:
    """
    The (multi-variate) normal distribution.

    The Gaussian distribution is ubiquitous in probability
    theory, since it is the final and stable or equilibrium
    distribution to which other distributions gravitate
    under a wide variety of smooth operations, e.g.,
    convolutions and stochastic transformations. One
    example of this is the central limit theorem. The
    Gaussian distribution is also attractive from a
    numerical point of view as it is maintained through
    many transformations (e.g. it is stable).

    Parameters
    ----------
    mean : float or array-like or LinearOperator
        Mean of the normal distribution.

    cov : float or array-like or LinearOperator
        (Co-)variance of the normal distribution.

    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to
        use for drawing realizations from this
        distribution. If None (or np.random), the global
        np.random state is used. If integer, it is used to
        seed the local
        :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    Distribution : Class representing general probability distributions.

    Examples
    --------
    >>> from probnum.prob import Normal
    >>> N = Normal(mean=0.5, cov=1.0)
    >>> N.parameters
    {'mean': 0.5, 'cov': 1.0}

    Todo
    ----
    Only keep Cholesky factors as covariance to avoid
    losing symmetry
    """

    def __new__(cls, mean=0., cov=1., random_state=None):
        """
        Factory method for normal subclasses.

        Checks shape/type of mean and cov and returns the
        corresponding type of Normal distribution:
            * _UnivariateNormal
            * _MultivariateNormal
            * _SymmetricKroneckerIdenticalFactorsNormal
            * _OperatorvariateNormal
        If neither applies, a ValueError is raised.
        """
        if cls is Normal:
            if _both_are_univariate(mean, cov):
                return _UnivariateNormal(mean, cov, random_state)
            elif _both_are_multi_or_matrixvariate(mean, cov):
                if len(mean.shape) == 1:
                    return _MultivariateNormal(mean, cov, random_state)
                else:
                    return _MatrixvariateNormal(mean, cov, random_state)
            elif _both_are_operatorvariate(mean, cov):
                if isinstance(cov, linops.SymmetricKronecker) and cov._ABequal:
                    return _SymmetricKroneckerIdenticalFactorsNormal(mean, cov, random_state)
                else:
                    return _OperatorvariateNormal(mean, cov, random_state)
            else:
                errmsg = ("Cannot instantiate normal distribution with mean of"
                          + "type {} and".format(mean.__class__.__name__)
                          + "covariance of"
                          + "type {}.".format(cov.__class__.__name__))
                raise ValueError(errmsg)
        else:
            return super(Normal, cls).__new__(cls, mean=mean, cov=cov,
                                              random_state=random_state)

def _both_are_univariate(mean, cov):
    """
    Checks whether mean and covar correspond to the
    UNIVARIATE normal distribution.
    """
    both_are_scalars = np.isscalar(mean) and np.isscalar(cov)
    mean_shape_dim1 = np.shape(mean) in [(1, 1), (1,), ()]
    cov_shape_dim1 = np.shape(cov) in [(1, 1), (1,), ()]
    both_in_dim1shapes = mean_shape_dim1 and cov_shape_dim1
    if both_are_scalars or both_in_dim1shapes:
        return True
    else:
        return False


def _both_are_multi_or_matrixvariate(mean, cov):
    """
    Checks whether mean and covar correspond to the
    MULTI- or MATRIXVARIATE normal distribution.
    """
    mean_is_multivar = isinstance(mean, (np.ndarray, scipy.sparse.spmatrix,))
    cov_is_multivar = isinstance(cov, (np.ndarray, scipy.sparse.spmatrix,))
    if mean_is_multivar and cov_is_multivar:
        return True
    else:
        return False

def _both_are_operatorvariate(mean, cov):
    """
    Checks whether mean OR (!) covar correspond to the
    OPERATORVARIATE normal distribution.
    """
    mean_is_opvariate = isinstance(mean, scipy.sparse.linalg.LinearOperator)
    cov_is_opvariate = isinstance(cov, scipy.sparse.linalg.LinearOperator)
    if mean_is_opvariate or cov_is_opvariate:
        return True
    else:
        return False