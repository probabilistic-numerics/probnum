"""Random variables represent in- and outputs of probabilistic numerical methods."""

import scipy.stats
import numpy as np


class RandomVariable:
    """
    Random variables are the main objects used by probabilistic numerical methods.

    In ``probnum`` every input is treated as a random variable even though most have Dirac or Gaussian measure. Each
    probabilistic numerical method takes a random variable encoding the prior distribution as input and outputs a
    random variable whose distribution encodes the uncertainty arising from finite computation. The generic signature
    of a probabilistic numerical method is:

    ``output_rv = probnum_method(input_rv, method_params)``

    Instances of :class:`RandomVariable` can be added, multiplied, etc. in a similar manner to vectors or linear
    operators, however depending on their distribution the result might not admit all methods. When creating a new
    subclass implementing a certain distribution, these operations should be overridden to represent the properties of
    the distribution.

    Parameters
    ----------

    See Also
    --------
    asrandomvariable : Transform into a RandomVariable.

    Examples
    --------
    """

    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov  # TODO: variance or covariance here?
        # TODO: add some type checking
        # TODO: allow construction from scipy distribution object
        raise NotImplementedError()

    # TODO: implement addition and multiplication with constant matrices / vectors
    # Example of spmatrix class: https://github.com/scipy/scipy/blob/v0.19.0/scipy/sparse/base.py#L62-L1108

    def pdf(self, x):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def cdf(self, x):
        raise NotImplementedError()

    def sample(self, size=1):
        """
        Returns realizations from the associated random variable.

        Parameters
        ----------
        size : tuple, default=1
            Shape of the realizations.

        Returns
        -------

        """
        raise NotImplementedError()


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


class Normal(RandomVariable):

    def __init__(self, mean=0, cov=1):
        # todo: allow for linear operators as mean and covariance
        super().__init__(mean=mean, cov=cov)

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.mean, cov=self.cov)

    def sample(self, size=1):
        return np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=size)
