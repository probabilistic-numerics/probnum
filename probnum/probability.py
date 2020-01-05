"""Random variables represent in- and outputs of probabilistic numerical methods."""

import scipy.stats
import numpy as np
from .linalg.linear_operators import LinearOperator

__all__ = ["RandomVariable", "DiracRV", "NormalRV", "asrandomvariable"]


class RandomVariable:
    """
    Random variables are the main objects used by probabilistic numerical methods.

    In ``probnum`` every input is treated as a random variable even though most have Dirac or Gaussian measure. Every
    probabilistic numerical method takes a random variable encoding the prior distribution as input and outputs a
    random variable whose distribution encodes the uncertainty arising from finite computation. The generic signature
    of a probabilistic numerical method is:

    ``output_rv = probnum_method(input_rv, method_params)``

    Instances of :class:`RandomVariable` can be added, multiplied, etc. in a similar manner to vectors or linear
    operators, however depending on their distribution the result might not admit all previously available methods
    anymore. When creating a new subclass implementing a certain distribution, these operations should be overridden to
    represent the properties of the distribution.

    Parameters
    ----------

    See Also
    --------
    asrandomvariable : Transform into a RandomVariable.

    Examples
    --------
    """

    def __new__(cls, *args, **kwargs):
        if cls is RandomVariable:
            # Operate as _CustomRandomVariable factory.
            return super(RandomVariable, cls).__new__(_CustomRandomVariable)
        else:
            # Run constructor of the corresponding class.
            return super(RandomVariable, cls).__new__(cls)

    # Todo: See RandomVariable in tensorflow
    # Todo: Attribute sample_type for operator overloading; dtype as init arg needed?
    def __init__(self, mean=None, cov=None, shape=None):
        self.mean = mean
        self.cov = cov
        # Set shape
        self._shape = None
        if mean is not None and shape is not None:
            if mean.shape != shape:
                raise ValueError("Shape of mean and given shape do not match.")
        elif shape is not None:
            self._get_shape(shape)
        else:
            self._get_shape(mean.shape)
        # TODO: add some type checking
        # TODO: allow construction from scipy distribution object

    # TODO: implement addition and multiplication with constant matrices / vectors
    # Example of spmatrix class: https://github.com/scipy/scipy/blob/v0.19.0/scipy/sparse/base.py#L62-L1108

    def _set_shape(self, shape):
        """
        Set the shape of a random variable.

        See Also
        --------
        reshape : Give a new shape to a random variable.
        """
        shape = tuple(shape)

        try:
            shape = int(shape[0]), int(shape[1])  # floats, other weirdness
        except:
            raise TypeError('Invalid shape.')

        if not all([i >= 0 for i in shape]):
            raise ValueError('Invalid shape.')

        if (self._shape != shape) and (self._shape is not None):
            try:
                self = self.reshape(shape)
            except NotImplementedError:
                raise NotImplementedError("Reshaping not implemented for %s." %
                                          self.__class__.__name__)
        self._shape = shape

    def _get_shape(self):
        """If applicable, return the shape of the random variable."""
        return self._shape

    shape = property(fget=_get_shape, fset=_set_shape)

    def reshape(self, shape):
        """
        Give a new shape to a random variable.

        Parameters
        ----------
        shape : tuple of ints
            The new shape should be compatible with the original shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.

        See Also
        --------
        np.matrix.reshape : NumPy's `'reshape'` for matrices.
        """
        raise NotImplementedError("Reshaping not implemented for {}.".format(self.__class__.__name__))

    def pdf(self, x):
        """
        Probability density function.

        Parameters
        ----------
        x

        Returns
        -------

        """
        raise NotImplementedError

    def logpdf(self, x):
        """
        Natural logarithm of the probability density function.

        Parameters
        ----------
        x

        Returns
        -------

        """
        raise NotImplementedError

    def cdf(self, y):
        """
        Cumulative distribution function.

        Parameters
        ----------
        y

        Returns
        -------

        """
        raise NotImplementedError

    def sample(self, size=1, seed=None):
        """
        Returns realizations from the associated random variable.

        Parameters
        ----------
        size : tuple, default=1
            Shape of the realizations.

        Returns
        -------

        """
        raise NotImplementedError


class _CustomRandomVariable(RandomVariable):
    """
    User-defined random variable via specified attributes and methods.
    """
    # TODO implement this similar to scipy's _CustomLinearOperator


class DiracRV(RandomVariable):
    """

    See Also
    --------
    RandomVariable : Random variables are the main in- and outputs of probabilistic numerical methods.
    """

    def __init__(self, mean):
        super().__init__(mean=mean)

    def sample(self, size=1):
        if size == 1:
            return self.mean
        else:
            return self.mean * np.ones(shape=size)


class NormalRV(RandomVariable):
    """

    See Also
    --------
    RandomVariable : Random variables are the main in- and outputs of probabilistic numerical methods.
    """

    def __init__(self, mean=0, cov=1):
        # todo: allow for linear operators as mean and covariance
        super().__init__(mean=mean, cov=cov)

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def cdf(self, y):
        return scipy.stats.multivariate_normal.cdf(y, mean=self.mean, cov=self.cov)

    def sample(self, size=1):
        return np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=size)


def asrandomvariable(X):
    """
    Return `X` as a :class:`RandomVariable`.

    Parameters
    ----------
    X : array-like or LinearOperator or scipy.stats.rv_continuous or scipy.stats.rv_discrete
        Argument to be represented as a random variable.

    Returns
    -------
    X : RandomVariable
        X as a random variable.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Examples
    --------
    >>> from probnum.probability import asrandomvariable
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> asrandomvariable(M)
    <2x3 RandomVariable with dtype=int32>
    """
    if isinstance(X, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
        # TODO: transform scipy distribution objects and numpy arrays automatically
        raise NotImplementedError
    elif isinstance(X, np.ndarray):
        raise NotImplementedError
    elif isinstance(X, scipy.sparse.linalg.LinearOperator):
        raise NotImplementedError
    else:
        raise ValueError("Argument of type {} cannot be converted to a random variable.".format(type(X)))
