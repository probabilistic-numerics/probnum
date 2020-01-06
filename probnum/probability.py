"""Random variables represent in- and outputs of probabilistic numerical methods."""

import scipy.stats
from scipy._lib._util import check_random_state
import numpy as np

# from .linalg.linear_operators import LinearOperator

__all__ = ["RandomVariable", "Distribution", "Dirac", "Normal", "asrandomvariable", "asdistribution"]


class RandomVariable:
    """
    Random variables are the main objects used by probabilistic numerical methods.

    In ``probnum`` in- and outputs are treated as random variables even though most have Dirac or Gaussian measure.
    Every probabilistic numerical method takes a random variable encoding the prior distribution as input and outputs a
    random variable whose distribution encodes the uncertainty arising from finite computation. The generic signature
    of a probabilistic numerical method is:

    ``output_rv = probnum_method(input_rv, method_params)``

    Instances of :class:`RandomVariable` can be added, multiplied, etc. in a similar manner to arrays or linear
    operators, however depending on their ``distribution`` the result might not admit all previously available methods
    anymore.

    Parameters
    ----------
    shape : tuple
        Shape of realizations of this random variable.
    dtype : str or numpy.dtype
        ``Dtype`` of realizations of this random variable.
    distribution : Distribution
        Probability distribution of the random variable.

    See Also
    --------
    asrandomvariable : Transform into a :class:`RandomVariable`.
    Distribution : A class representing probability distributions.

    Examples
    --------
    """

    def __init__(self, shape=(), dtype=None, distribution=None):
        """Create a new random variable."""
        # Set shape and dtype
        self._shape = shape
        self._dtype = dtype
        if dtype is not None:
            if isinstance(dtype, np.dtype):
                self._dtype = dtype
            else:
                self._dtype = np.dtype(dtype)
        if distribution.mean is not None:
            if distribution.mean.shape != shape:
                raise ValueError("Shape of distribution mean and given shape do not match.")
            else:
                self._shape = distribution.mean.shape
            self._dtype = distribution.mean.dtype
        # Set distribution of random variable
        if distribution is not None:
            self._distribution = Distribution(distribution)
        else:
            self._distribution = Distribution(mean=distribution.mean, cov=distribution.cov, sample=distribution.sample)
        # TODO: add some type checking (e.g. for shape as a tuple of ints)

    @property
    def distribution(self):
        """Distribution of random variable."""
        return self._distribution

    @property
    def mean(self):
        """Expected value of the random variable."""
        if self._distribution is not None:
            try:
                return self._distribution.parameters["mean"]
            except KeyError:
                raise NotImplementedError("Underlying {} has no mean.".format(type(self._distribution).__name__))
        else:
            raise NotImplementedError("No underlying distribution specified.")

    @property
    def cov(self):
        """Covariance operator of the random variable"""
        if self._distribution is not None:
            try:
                return self._distribution.parameters["cov"]
            except KeyError:
                raise NotImplementedError("Underlying {} has no covariance.".format(type(self._distribution).__name__))
        else:
            raise NotImplementedError("No underlying distribution specified.")

    @property
    def shape(self):
        """Shape of realizations of the random variable."""
        return self._shape

    @property
    def dtype(self):
        """`Dtype` of elements in this random variable."""
        return self._dtype

    @property
    def random_state(self):
        """Random state of the random variable induced by its ``distribution``."""
        return self._distribution.random_state

    @random_state.setter
    def random_state(self, seed):
        """ Get or set the RandomState object of the underlying distribution.

        This can be either None or an existing :class:`~numpy.random.RandomState` object.
        If None (or np.random), use the :class:`~numpy.random.RandomState` singleton used by np.random.
        If already a :class:`~numpy.random.RandomState` instance, use it.
        If an int, use a new :class:`~numpy.random.RandomState` instance seeded with seed.
        """
        self.distribution._random_state = check_random_state(seed)

    # TODO: implement addition and multiplication with constant matrices / vectors
    # Example of spmatrix class: https://github.com/scipy/scipy/blob/v0.19.0/scipy/sparse/base.py#L62-L1108

    def sample(self, size=()):
        """
        Draw realizations from a random variable.

        Parameters
        ----------
        size : tuple
            Size of the drawn sample of realizations.

        Returns
        -------
        sample : array-like
            Sample of realizations with the given ``size`` and the inherent ``shape``.
        """
        if self._distribution is not None:
            return self._distribution.sample(size=size)
        else:
            raise NotImplementedError("No sampling method provided.")

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
        """
        raise NotImplementedError("Reshaping not implemented for {}.".format(self.__class__.__name__))


class Distribution:
    """
    A class representing probability distributions.

    When creating a subclass implementing a certain distribution, generic operations (addition, multiplication, ...)
     should be overridden to represent the properties of the distribution.

    Parameters
    ----------
    distparams : dict
        Dictionary of distribution parameters such as mean, variance et cetera.
    pdf : function
        Probability density or mass function.
    logpdf : function
        Log-probability density or mass function.
    cdf : function
        Cumulative distribution function.
    cdf : function
        Log-cumulative distribution function.
    sample : function
        Function implementing sampling. Must have signature ``sample(size=())``.
    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to use for drawing
        realizations from this distribution.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    asdistribution : Transform object into a :class:`Distribution`.
    RandomVariable : Random variables are the main objects used by probabilistic numerical methods.

    Examples
    --------

    """

    def __init__(self, distparams=None, pdf=None, logpdf=None, cdf=None, logcdf=None, sample=None, random_state=None):
        self._parameters = distparams
        self._pdf = pdf
        self._logpdf = logpdf
        self._cdf = cdf
        self._logcdf = logcdf
        self._sample = sample
        self._random_state = check_random_state(random_state)
        # TODO: allow construction from scipy distribution object

    @property
    def random_state(self):
        """Random state of the distribution."""
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        """ Get or set the RandomState object of the underlying distribution.

        This can be either None or an existing RandomState object.
        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.
        """
        self._random_state = check_random_state(seed)

    @property
    def parameters(self):
        """Parameters of the probability distribution."""
        if self._parameters is not None:
            return self._parameters
        else:
            raise NotImplementedError("No parameters of {} are available.".format(type(self).__name__))

    def _check_distparams(self):
        pass
        # TODO: type checking of self.parameters

    def pdf(self, x):
        """
        Probability density or mass function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the probability density/mass function.

        Returns
        -------
        p : array-like
            Value of the probability density / mass function at the given points.

        """
        if self._pdf is not None:
            return self._pdf(x)
        if self._logpdf is not None:
            return np.exp(self._logpdf(x))
        raise NotImplementedError('The function \'pdf\' is not implemented for {}'.format(type(self).__name__))

    def logpdf(self, x):
        """
        Natural logarithm of the probability density function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the log-probability density/mass function.

        Returns
        -------
        logp : array-like
            Value of the log-probability density / mass function at the given points.
        """
        if hasattr(self, '_logpdf'):
            return self._logpdf(x)
        if hasattr(self, '_pdf'):
            return np.log(self._pdf(x))
        raise NotImplementedError('The function \'logpdf\' is not implemented for {}'.format(type(self).__name__))

    def cdf(self, y):
        """
        Cumulative distribution function.

        Parameters
        ----------
        y : array-like
            Evaluation points of the cumulative distribution function.

        Returns
        -------
        x : array-like
            Value of the cumulative density function at the given points.
        """
        if self._cdf is not None:
            return self._cdf(y)
        if self._logcdf is not None:
            return np.exp(self._logcdf(y))
        raise NotImplementedError('The function \'cdf\' is not implemented for {}'.format(type(self).__name__))

    def logcdf(self, y):
        """
        Log-cumulative distribution function.

        Parameters
        ----------
        y : array-like
            Evaluation points of the cumulative distribution function.

        Returns
        -------
        x : array-like
            Value of the log-cumulative density function at the given points.
        """
        if self._logcdf is not None:
            return self._logpdf(y)
        if self._cdf is not None:
            return np.log(self._cdf(y))
        raise NotImplementedError('The function \'logcdf\' is not implemented for {}'.format(type(self).__name__))

    def sample(self, size=()):
        """
        Returns realizations from the associated random variable.

        Parameters
        ----------
        size : tuple, default=()
            Shape of the realizations.

        Returns
        -------

        """
        if self._sample is not None:
            return self._sample(size=size)
        raise NotImplementedError('The function \'sample\' is not implemented for {}'.format(type(self).__name__))


class Dirac(Distribution):
    """
    The Delta or Dirac distribution.

    See Also
    --------
    Distribution : Class implementing probability distributions.
    """

    def __init__(self, mean):
        super().__init__(distparams={"mean": mean})

    def sample(self, size=(), seed=None):
        if size == 1:
            return self._parameters["mean"]
        else:
            return self._parameters["mean"] * np.ones(shape=size)


class Normal(Distribution):
    """
    The multi-variate normal distribution.

    See Also
    --------
    Distribution : Class implementing probability distributions.
    """

    def __init__(self, mean=0, cov=1):
        # todo: allow for linear operators as mean and covariance
        super().__init__(distparams={"mean": mean, "cov": cov})

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def cdf(self, y):
        return scipy.stats.multivariate_normal.cdf(y, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def sample(self, size=1, seed=None):
        return np.random.multivariate_normal(mean=self.parameters["mean"], cov=self.parameters["cov"], size=size)


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


def asdistribution(dist):
    """
    Return ``dist`` as a :class:`Distribution`.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous or scipy.stats.rv_discrete or object
        Argument to be represented as a :class:`Distribution`.

    Returns
    -------
    dist : Distribution
        The object `dist` as a :class:`Distribution`.

    See Also
    --------
    Distribution : A class representing probability distributions.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> from probnum.probability import asdistribution
    >>> d = asdistribution(bernoulli)
    >>> d.sample()

    """
    if isinstance(dist, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
        raise NotImplementedError
    else:
        raise NotImplementedError
