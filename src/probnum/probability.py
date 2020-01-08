"""Random variables represent in- and outputs of probabilistic numerical methods."""

import numpy as np
import operator
import warnings
import scipy.stats
from scipy._lib._util import check_random_state

__all__ = ["RandomVariable", "Distribution", "Dirac", "Normal", "asrandomvariable", "asdistribution"]


class RandomVariable:
    """
    Random variables are the main objects used by probabilistic numerical methods.

    Every probabilistic numerical method takes a random variable encoding the prior distribution as input and outputs a
    random variable whose distribution encodes the uncertainty arising from finite computation. The generic signature
    of a probabilistic numerical method is:

    ``output_rv = probnum_method(input_rv, method_params)``

    In practice, most random variables used by methods in ProbNum have Dirac or Gaussian measure.

    Instances of :class:`RandomVariable` can be added, multiplied, etc. in a similar manner to arrays or linear
    operators, however depending on their ``distribution`` the result might not admit all previously available methods
    anymore.

    Parameters
    ----------
    shape : tuple
        Shape of realizations of this random variable.
    dtype : numpy.dtype or object
        Data type of realizations of this random variable. If ``object`` will be converted to ``numpy.dtype``.
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
            self._distribution = asdistribution(dist=distribution)
        else:
            self._distribution = Distribution()
        # TODO: add some type checking (e.g. for shape as a tuple of ints)

    @property
    def distribution(self):
        """Distribution of random variable."""
        return self._distribution

    def mean(self):
        """Expected value of the random variable."""
        if self._distribution is not None:
            try:
                return self._distribution.mean()
            except KeyError:
                raise NotImplementedError("Underlying {} has no mean.".format(type(self._distribution).__name__))
        else:
            raise NotImplementedError("No underlying distribution specified.")

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

    # Arithmetic operations
    def _get_new_attr(self, attr, other, op):
        """
        Get new shared attribute of self and other.

        Parameters
        ----------
        attr : str
            Attribute to be combined.
        other : RandomVariable
            Other random variable to combine with ``self``.

        Returns
        -------
        value : object
            Shared attribute value.

        """
        self_attr = getattr(self, attr)
        other_attr = getattr(other, attr)
        if self_attr is None or other_attr is None:
            return None
        elif self_attr != other_attr:
            try:
                # infer attribute (shape, dtype, ...) from broadcasting with means
                return getattr(op(self.mean(), other.mean()), attr)
            except (ValueError, NotImplementedError):
                try:
                    # infer attribute (shape, dtype, ...) from broadcasting with sampling
                    self.random_state = check_random_state(1)
                    other.random_state = check_random_state(1)
                    warnings.warn(
                        "Attributes of combined random variable inferred through a sample. This might adversely" +
                        " affect performance. Consider matching attributes (shape, dtype, ...) before combining.")
                    return getattr(op(self.sample(size=1), other.sample(size=1)), attr)
                except (ValueError, NotImplementedError):
                    raise ValueError(
                        "ValueError: Objects {} does not match or cannot be broadcast together.".format(attr))
        else:
            return self_attr

    # Binary arithmetic operations
    def _rv_from_op(self, other, op):
        """
        Create a new random variable by applying a binary operation.

        Parameters
        ----------
        other : object
        op : function
            Binary operation.

        Returns
        -------
        combined_randvar : RandomVariable
            Random variable resulting from ``op``.

        """
        other_rv = asrandomvariable(other)
        return RandomVariable(shape=self._get_new_attr(attr="shape", other=other_rv, op=op),
                              dtype=self._get_new_attr(attr="dtype", other=other_rv, op=op),
                              distribution=op(self.distribution, other_rv.distribution))

    def __add__(self, other):
        return self._rv_from_op(other=other, op=operator.add)

    def __sub__(self, other):
        return self._rv_from_op(other=other, op=operator.sub)

    def __mul__(self, other):
        return self._rv_from_op(other=other, op=operator.mul)

    def __matmul__(self, other):
        return self._rv_from_op(other=other, op=operator.matmul)

    def __truediv__(self, other):
        return self._rv_from_op(other=other, op=operator.truediv)

    def __pow__(self, power, modulo=None):
        return self._rv_from_op(other=power, op=operator.pow)

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        other_rv = asrandomvariable(other)
        return other_rv._rv_from_op(other=self, op=operator.add)

    def __rsub__(self, other):
        other_rv = asrandomvariable(other)
        return other_rv._rv_from_op(other=self, op=operator.sub)

    def __rmul__(self, other):
        other_rv = asrandomvariable(other)
        return other_rv._rv_from_op(other=self, op=operator.mul)

    def __rmatmul__(self, other):
        other_rv = asrandomvariable(other)
        return other_rv._rv_from_op(other=self, op=operator.matmul)

    def __rtruediv__(self, other):
        other_rv = asrandomvariable(other)
        return other_rv._rv_from_op(other=self, op=operator.truediv)

    def __rpow__(self, power, modulo=None):
        other_rv = asrandomvariable(power)
        return other_rv._rv_from_op(other=self, op=operator.pow)

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    def __iadd__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __imatmul__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    def __ipow__(self, power, modulo=None):
        raise NotImplementedError

    # Unary arithmetic operations
    def __neg__(self):
        return RandomVariable(shape=self.shape,
                              dtype=self.dtype,
                              distribution=operator.neg(self.distribution))

    def __pos__(self):
        return RandomVariable(shape=self.shape,
                              dtype=self.dtype,
                              distribution=operator.pos(self.distribution))

    def __abs__(self):
        return RandomVariable(shape=self.shape,
                              dtype=self.dtype,
                              distribution=operator.abs(self.distribution))


class Distribution:
    """
    A class representing probability distributions.

    This class is primarily intended to be subclassed to provide distribution-specific implementations of the various
    methods (logpdf, logcdf, sample, mean, var, ...). When creating a subclass implementing a certain distribution,
    make sure to override generic operations (addition, multiplication, ...) to represent the properties of the
    distribution. This allows algebraic operations on instances of :class:`Distribution` and :class:`RandomVariable`.

    Parameters
    ----------
    parameters : dict
        Dictionary of distribution parameters such as mean, variance, et cetera.
    pdf : function
        Probability density or mass function.
    logpdf : function
        Log-probability density or mass function.
    cdf : function
        Cumulative distribution function.
    logcdf : function
        Log-cumulative distribution function.
    sample : function
        Function implementing sampling. Must have signature ``sample(size=())``.
    mean : function
        Function returning the mean of the distribution.
    var : function
        Function returning the variance of the distribution.
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

    def __init__(self, parameters=None, pdf=None, logpdf=None, cdf=None, logcdf=None, sample=None,
                 mean=None, var=None, random_state=None):
        if parameters is None:
            parameters = {}  # sentinel value to avoid anti-pattern
        self._parameters = parameters
        self._pdf = pdf
        self._logpdf = logpdf
        self._cdf = cdf
        self._logcdf = logcdf
        self._sample = sample
        self._mean = mean
        self._var = var
        self._random_state = check_random_state(random_state)

    @property
    def random_state(self):
        """Random state of the distribution.

        This attribute defines the RandomState object to use for drawing
        realizations from this distribution.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState` instance.
        """
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
        """
        Parameters of the probability distribution.

        The parameters of the distribution such as mean, variance, et cetera stored in a ``dict``.
        """
        if self._parameters is not None:
            return self._parameters
        else:
            raise NotImplementedError("No parameters of {} are available.".format(type(self).__name__))

    def _check_distparams(self):
        pass
        # TODO: type checking of self._parameters; check method signatures.

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

    def cdf(self, x):
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the cumulative distribution function.

        Returns
        -------
        q : array-like
            Value of the cumulative density function at the given points.
        """
        if self._cdf is not None:
            return self._cdf(x)
        if self._logcdf is not None:
            return np.exp(self._logcdf(x))
        raise NotImplementedError('The function \'cdf\' is not implemented for {}'.format(type(self).__name__))

    def logcdf(self, x):
        """
        Log-cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the cumulative distribution function.

        Returns
        -------
        q : array-like
            Value of the log-cumulative density function at the given points.
        """
        if self._logcdf is not None:
            return self._logcdf(x)
        if self._cdf is not None:
            return np.log(self._cdf(x))
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

    def median(self):
        """
        Median of the distribution.

        Returns
        -------
        median : float
            The median of the distribution.
        """
        return self.cdf(x=0.5)

    def mean(self):
        """
        Mean :math:`\\mathbb{E}(X)` of the distribution.

        Returns
        -------
        mean : float
            The mean of the distribution.
        """
        if self._mean is not None:
            return self._mean()
        elif "mean" in self._parameters:
            return self._parameters["mean"]
        else:
            raise NotImplementedError('The function \'mean\' is not implemented for {}'.format(type(self).__name__))

    def var(self):
        """
        Variance :math:`\\operatorname{Var}(X) = \\mathbb{E}((X-\\mathbb{E}(X))^2)` of the distribution.

        Returns
        -------
        var : float
            The variance of the distribution.
        """
        if self._var is not None:
            return self._var()
        elif "var" in self._parameters:
            return self._parameters["var"]
        else:
            raise NotImplementedError('The function \'var\' is not implemented for {}'.format(type(self).__name__))

    def std(self):
        """
        Standard deviation of the distribution.

        Returns
        -------
        std : float
            The standard deviation of the distribution.
        """
        return np.sqrt(self.var())

    # Binary arithmetic operations
    def __add__(self, other):
        # todo: handle other being a number, etc. in asdistribution (avoids type checking in each overloaded method)
        otherdist = asdistribution(other)
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __rpow__(self, power, modulo=None):
        raise NotImplementedError

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    def __iadd__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __imatmul__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    def __ipow__(self, power, modulo=None):
        raise NotImplementedError

    # Unary arithmetic operations
    def __neg__(self):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __abs__(self):
        raise NotImplementedError


class Dirac(Distribution):
    """
    The Dirac delta function.

    This distribution models a point mass and can be useful to represent numbers as random variables with Dirac measure.
    It has the useful property that algebraic operations between a :class:`Dirac` random variable and an arbitrary
    :class:`RandomVariable` acts in the same way as the algebraic operation with a constant.

    Note, that a Dirac measure does not admit a probability density function but can be viewed as a distribution
    (generalized function).

    See Also
    --------
    Distribution : Class representing general probability distributions.
    """

    def __init__(self, support=0):
        super().__init__(parameters={"support": support})

    def cdf(self, x):
        if x < self.parameters["support"]:
            return 0
        else:
            return 1

    def median(self):
        return self.parameters["support"]

    def mean(self):
        return self.parameters["support"]

    def var(self):
        return 0

    def sample(self, size=(), seed=None):
        if size == 1:
            return self.parameters["support"]
        else:
            return self.parameters["support"] * np.ones(shape=size)


class Normal(Distribution):
    """
    The (multi-variate) normal distribution.

    The Gaussian distribution is ubiquitous in probability theory, since it is the final and stable or equilibrium
    distribution to which other distributions gravitate under a wide variety of smooth operations, e.g.,
    convolutions and stochastic transformations. One example of this is the central limit theorem. The Gaussian
    distribution is also attractive from a numerical point of view as it is maintained through many transformations
    (e.g. it is stable).

    See Also
    --------
    Distribution : Class representing general probability distributions.
    """

    def __init__(self, mean=0, cov=1):
        # todo: allow for linear operators as mean and covariance
        super().__init__(parameters={"mean": mean, "cov": cov})

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

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
        # TODO: allow construction from scipy distribution object
        raise NotImplementedError
    # Todo: allow construction from numbers / arrays to create dirac distribution
    else:
        raise NotImplementedError
