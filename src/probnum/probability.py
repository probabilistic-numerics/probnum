"""Random variables represent in- and outputs of probabilistic numerical methods."""

import operator

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.linalg import linear_operators

__all__ = ["RandomVariable", "Distribution", "Dirac", "Normal", "asrandvar", "asdist"]


class RandomVariable:
    """
    Random variables are the main objects used by probabilistic numerical methods.

    Every probabilistic numerical method takes a random variable encoding the prior distribution as input and outputs a
    random variable whose distribution encodes the uncertainty arising from finite computation. The generic signature
    of a probabilistic numerical method is:

    ``output_rv = probnum_method(input_rv, method_params)``

    In practice, most random variables used by methods in ProbNum have Dirac or Gaussian measure.

    Instances of :class:`RandomVariable` can be added, multiplied, etc. with arrays and linear operators. This may
    change their ``distribution`` and not necessarily all previously available methods are retained.

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

    def __init__(self, shape=None, dtype=None, distribution=None):
        """Create a new random variable."""
        # Set dtype (in accordance with distribution)
        self._dtype = dtype
        if dtype is not None:
            if isinstance(dtype, np.dtype):
                self._dtype = dtype
            else:
                self._dtype = np.dtype(dtype)
        if distribution is not None:
            if self.dtype is None:
                self._dtype = distribution.dtype
            elif self.dtype != distribution.dtype:
                # Change distribution dtype if random variable type is different
                distribution.dtype = dtype

        # Set shape (in accordance with distribution mean)
        self._shape = shape
        if distribution is not None:
            if distribution.mean is not None:
                if np.isscalar(distribution.mean()):
                    shape_mean = ()
                else:
                    shape_mean = distribution.mean().shape
                if shape is None or shape_mean == shape:
                    self._shape = shape_mean
                else:
                    raise ValueError("Shape of distribution mean and given shape do not match.")

        # Set distribution of random variable
        if distribution is not None:
            self._distribution = asdist(obj=distribution)
        else:
            self._distribution = Distribution()
        # TODO: add some type checking (e.g. for shape as a tuple of ints) and extract as function
        # TODO: Extract dtype and shape checking as a function

    def __repr__(self):
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)
        return '<%s %s with %s>' % (str(self.shape), self.__class__.__name__, dt)

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
        self.distribution._random_state = scipy._lib._util.check_random_state(seed)

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
        shape : int or tuple of ints
            New shape for the random variable. It must be compatible with the original shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.
        """
        # Set shape
        self._shape = shape

        # Change distribution parameters
        self._distribution.reshape(shape=shape)

    # Binary arithmetic operations

    # The below prevents numpy from calling elementwise arithmetic operations allowing expressions like:
    # y = np.array([1, 1]) + RV
    # to call the arithmetic operations defined by RandomVariable instead of elementwise. Thus no
    # array of RandomVariables but a RandomVariable with the correct shape is returned.
    __array_ufunc__ = None

    def _rv_from_op(self, other, op):
        """
        Create a new random variable by applying a binary operation.

        Parameters
        ----------
        other : object
        op : callable
            Binary operation.

        Returns
        -------
        combined_rv : RandomVariable
            Random variable resulting from ``op``.

        """
        other_rv = asrandvar(other)
        combined_rv = RandomVariable(distribution=op(self.distribution, other_rv.distribution))
        return combined_rv

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
        other_rv = asrandvar(other)
        return other_rv._rv_from_op(other=self, op=operator.add)

    def __rsub__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_op(other=self, op=operator.sub)

    def __rmul__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_op(other=self, op=operator.mul)

    def __rmatmul__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_op(other=self, op=operator.matmul)

    def __rtruediv__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_op(other=self, op=operator.truediv)

    def __rpow__(self, power, modulo=None):
        other_rv = asrandvar(power)
        return other_rv._rv_from_op(other=self, op=operator.pow)

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    # TODO: needs setter functions for properties `shape` and `dtype` to do in place
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
    methods (``logpdf``, ``logcdf``, ``sample``, ``mean``, ``var``, ...). When creating a subclass implementing a
    certain distribution, make sure to override generic operations (addition, multiplication, ...) to represent the
    properties of the distribution. The overriden methods should also update the ``parameters`` of the distribution.
    This allows arithmetic operations on instances of :class:`Distribution` and :class:`RandomVariable`.

    Parameters
    ----------
    parameters : dict
        Dictionary of distribution parameters such as mean, variance, et cetera.
    pdf : callable
        Probability density or mass function.
    logpdf : callable
        Log-probability density or mass function.
    cdf : callable
        Cumulative distribution function.
    logcdf : callable
        Log-cumulative distribution function.
    sample : callable
        Function implementing sampling. Must have signature ``sample(size=())``.
    mean : callable
        Function returning the mean of the distribution.
    var : callable
        Function returning the variance of the distribution.
    dtype : numpy.dtype or object
        Data type of realizations of a random variable with this distribution. If ``object`` will be converted to ``numpy.dtype``.
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
                 mean=None, var=None, dtype=None, random_state=None):
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
        self._dtype = dtype
        self._random_state = scipy._lib._util.check_random_state(random_state)

    @property
    def dtype(self):
        """`Dtype` of elements of samples from this distribution."""
        return self._dtype

    @dtype.setter
    def dtype(self, newtype):
        """Set the `dtype` of the distribution."""
        self._dtype = newtype

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
        self._random_state = scipy._lib._util.check_random_state(seed)

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
        raise NotImplementedError(
            'The function \'pdf\' is not implemented for object of class {}'.format(type(self).__name__))

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
        raise NotImplementedError(
            'The function \'logpdf\' is not implemented for object of class {}'.format(type(self).__name__))

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
        raise NotImplementedError(
            'The function \'cdf\' is not implemented for object of class {}'.format(type(self).__name__))

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
        raise NotImplementedError(
            'The function \'logcdf\' is not implemented for object of class {}'.format(type(self).__name__))

    def sample(self, size=()):
        """
        Returns realizations from the associated random variable.

        Parameters
        ----------
        size : tuple, default=()
            Size of the realizations.

        Returns
        -------
        realizations : array-like or LinearOperator
            Realizations of the underlying random variable.
        """
        if self._sample is not None:
            return self._sample(size=size)
        raise NotImplementedError(
            'The function \'sample\' is not implemented for object of class {}'.format(type(self).__name__))

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
        mean : array-like
            The mean of the distribution.
        """
        if self._mean is not None:
            return self._mean()
        elif "mean" in self._parameters:
            return self._parameters["mean"]
        else:
            raise NotImplementedError(
                'The function \'mean\' is not implemented for object of class {}'.format(type(self).__name__))

    def var(self):
        """
        Variance :math:`\\operatorname{Var}(X) = \\mathbb{E}((X-\\mathbb{E}(X))^2)` of the distribution.

        Returns
        -------
        var : array-like
            The variance of the distribution.
        """
        if self._var is not None:
            return self._var()
        elif "var" in self._parameters:
            return self._parameters["var"]
        else:
            raise NotImplementedError(
                'The function \'var\' is not implemented for object of class {}'.format(type(self).__name__))

    def std(self):
        """
        Standard deviation of the distribution.

        Returns
        -------
        std : array-like
            The standard deviation of the distribution.
        """
        return np.sqrt(self.var())

    def reshape(self, shape):
        """
        Give a new shape to (realizations of) this distribution.

        Parameters
        ----------
        shape : int or tuple of ints
            New shape for the realizations and parameters of this distribution. It must be compatible with the original
            shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.
        """
        raise NotImplementedError(
            "Reshaping not implemented for distribution of type: {}.".format(self.__class__.__name__))

    # Binary arithmetic operations
    def __add__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            return Distribution(parameters={},  # correct updates of parameters should be handled in subclasses
                                pdf=lambda x: self.pdf(x - delta),
                                logpdf=lambda x: self.logpdf(x - delta),
                                cdf=lambda x: self.cdf(x - delta),
                                logcdf=lambda x: self.logcdf(x - delta),
                                sample=lambda size: self.sample(size=size) + delta,
                                mean=lambda: self.mean() + delta,
                                var=self.var,
                                random_state=self.random_state)
        else:
            raise NotImplementedError(
                "Addition not implemented for {} and {}.".format(self.__class__.__name__, other.__class__.__name__))

    def __sub__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return self + (-otherdist)
        else:
            raise NotImplementedError(
                "Subtraction not implemented for {} and {}.".format(self.__class__.__name__, other.__class__.__name__))

    def __mul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            if delta == 0:
                return Dirac(support=0 * self.mean(), random_state=self.random_state)
            else:
                return Distribution(parameters={},
                                    pdf=lambda x: self.pdf(x / delta),
                                    logpdf=lambda x: self.logpdf(x / delta),
                                    cdf=lambda x: self.cdf(x / delta),
                                    logcdf=lambda x: self.logcdf(x / delta),
                                    sample=lambda size: self.sample(size=size) * delta,
                                    mean=lambda: self.mean() * delta,
                                    var=lambda: self.var() * delta ** 2,
                                    random_state=self.random_state)
        else:
            raise NotImplementedError(
                "Multiplication not implemented for {} and {}.".format(self.__class__.__name__,
                                                                       other.__class__.__name__))

    def __matmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            return Distribution(parameters={},
                                sample=lambda size: self.sample(size=size) @ delta,
                                mean=lambda: self.mean() @ delta,
                                var=delta @ self.var @ delta.transpose(),
                                random_state=self.random_state)
        raise NotImplementedError(
            "Matrix multiplication not implemented for {} and {}.".format(self.__class__.__name__,
                                                                          other.__class__.__name__))

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division by zero not supported.")
        else:
            otherdist = asdist(other)
            if isinstance(otherdist, Dirac):
                return self * operator.inv(otherdist)
            else:
                raise NotImplementedError(
                    "Division not implemented for {} and {}.".format(self.__class__.__name__, other.__class__.__name__))

    def __pow__(self, power, modulo=None):
        raise NotImplementedError(
            "Exponentiation not implemented for {} and {}.".format(self.__class__.__name__, power.__class__.__name__))

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return self + otherdist
        else:
            raise NotImplementedError(
                "Addition not implemented for {} and {}.".format(other.__class__.__name__, self.__class__.__name__))

    def __rsub__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return operator.neg(self) + otherdist
        else:
            raise NotImplementedError(
                "Subtraction not implemented for {} and {}.".format(other.__class__.__name__, self.__class__.__name__))

    def __rmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return self * otherdist
        else:
            raise NotImplementedError(
                "Multiplication not implemented for {} and {}.".format(other.__class__.__name__,
                                                                       self.__class__.__name__))

    def __rmatmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            return Distribution(parameters={},
                                sample=lambda size: delta @ self.sample(size=size),
                                mean=lambda: delta @ self.mean,
                                var=delta @ (self.var @ delta.transpose()),
                                random_state=self.random_state)
        raise NotImplementedError(
            "Matrix multiplication not implemented for {} and {}.".format(other.__class__.__name__,
                                                                          self.__class__.__name__))

    def __rtruediv__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return operator.inv(self) * otherdist
        else:
            raise NotImplementedError(
                "Division not implemented for {} and {}.".format(other.__class__.__name__, self.__class__.__name__))

    def __rpow__(self, power, modulo=None):
        raise NotImplementedError(
            "Exponentiation not implemented for {} and {}.".format(power.__class__.__name__, self.__class__.__name__))

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
        try:
            return Distribution(parameters={},  # correct updates of parameters should be handled in subclasses
                                sample=lambda size: -self.sample(size=size),
                                mean=lambda: -self.mean(),
                                var=self.var,
                                random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Negation not implemented for {}.".format(self.__class__.__name__))

    def __pos__(self):
        try:
            return Distribution(parameters={},  # correct updates of parameters should be handled in subclasses
                                sample=lambda size: operator.pos(self.sample(size=size)),
                                mean=lambda: operator.pos(self.mean()),
                                var=self.var,
                                random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Negation not implemented for {}.".format(self.__class__.__name__))

    def __abs__(self):
        try:
            return Distribution(parameters={},  # correct updates of parameters should be handled in subclasses
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Absolute value not implemented for {}.".format(self.__class__.__name__))

    def __invert__(self):
        try:
            return Distribution(parameters={},  # correct updates of parameters should be handled in subclasses
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Inversion not implemented for {}.".format(self.__class__.__name__))


class Dirac(Distribution):
    """
    The Dirac delta distribution.

    This distribution models a point mass and can be useful to represent numbers as random variables with Dirac measure.
    It has the useful property that arithmetic operations between a :class:`Dirac` random variable and an arbitrary
    :class:`RandomVariable` acts in the same way as the arithmetic operation with a constant.

    Note, that a Dirac measure does not admit a probability density function but can be viewed as a distribution
    (generalized function).

    Parameters
    ----------
    support : scalar or array-like or LinearOperator
        The support of the dirac delta function.

    See Also
    --------
    Distribution : Class representing general probability distributions.

    Examples
    --------
    >>> from probnum.probability import RandomVariable, Dirac
    >>> dist1 = Dirac(support=0.)
    >>> dist2 = Dirac(support=1.)
    >>> rv = RandomVariable(distribution=dist1 + dist2)
    >>> rv.sample(size=5)
    array([1., 1., 1., 1., 1.])
    """

    def __init__(self, support, random_state=None):
        # Set dtype
        if np.isscalar(support):
            _dtype = np.dtype(type(support))
        else:
            _dtype = support.dtype

        # Initializer of superclass
        super().__init__(parameters={"support": support}, dtype=_dtype, random_state=random_state)

    def cdf(self, x):
        if x < self.parameters["support"]:
            return 0.
        else:
            return 1.

    def median(self):
        return self.parameters["support"]

    def mean(self):
        return self.parameters["support"]

    def var(self):
        return 0.

    def sample(self, size=(), seed=None):
        if size == 1:
            return self.parameters["support"]
        else:
            return np.full(fill_value=self.parameters["support"], shape=size)

    def reshape(self, shape):
        try:
            # Reshape support
            self._parameters["support"].reshape(shape=shape)
        except ValueError:
            raise ValueError("Cannot reshape this Dirac distribution to the given shape: {}".format(str(shape)))

    # Binary arithmetic operations
    def __add__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return Dirac(support=self.parameters["support"] + otherdist.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__add__(other=self)

    def __sub__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return Dirac(support=self.parameters["support"] - otherdist.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__rsub__(other=self)

    def __mul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return Dirac(support=self.parameters["support"] * otherdist.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__mul__(other=self)

    def __matmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return Dirac(support=self.parameters["support"] @ otherdist.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__rmatmul__(other=self)

    def __truediv__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return Dirac(support=operator.truediv(self.parameters["support"], otherdist.parameters["support"]),
                         random_state=self.random_state)
        else:
            return other.__rtruediv__(other=self)

    def __pow__(self, power, modulo=None):
        otherdist = asdist(power)
        if isinstance(otherdist, Dirac):
            return Dirac(support=pow(self.parameters["support"], otherdist.parameters["support"], modulo),
                         random_state=self.random_state)
        else:
            return power.__rpow__(power=self, modulo=modulo)

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        return asdist(other).__add__(other=self)

    def __rsub__(self, other):
        return asdist(other).__sub__(other=self)

    def __rmul__(self, other):
        return asdist(other).__mul__(other=self)

    def __rmatmul__(self, other):
        return asdist(other).__matmul__(other=self)

    def __rtruediv__(self, other):
        return asdist(other).__truediv__(other=self)

    def __rpow__(self, power, modulo=None):
        return asdist(power).__pow__(power=self)

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    def __iadd__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            self.parameters["support"] = self.parameters["support"] + otherdist.parameters["support"]
            return self
        else:
            raise NotImplementedError

    def __isub__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            self.parameters["support"] = self.parameters["support"] - otherdist.parameters["support"]
            return self
        else:
            raise NotImplementedError

    def __imul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            self.parameters["support"] = self.parameters["support"] * otherdist.parameters["support"]
            return self
        else:
            raise NotImplementedError

    def __imatmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            self.parameters["support"] = self.parameters["support"] @ otherdist.parameters["support"]
            return self
        else:
            raise NotImplementedError

    def __itruediv__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            self.parameters["support"] = operator.truediv(self.parameters["support"], otherdist.parameters["support"])
            return self
        else:
            raise NotImplementedError

    def __ipow__(self, power, modulo=None):
        otherdist = asdist(power)
        if isinstance(otherdist, Dirac):
            self.parameters["support"] = pow(self.parameters["support"], otherdist.parameters["support"], modulo)
            return self
        else:
            raise NotImplementedError

    # Unary arithmetic operations
    def __neg__(self):
        self.parameters["support"] = operator.neg(self.parameters["support"])
        return self

    def __pos__(self):
        self.parameters["support"] = operator.pos(self.parameters["support"])
        return self

    def __abs__(self):
        self.parameters["support"] = operator.abs(self.parameters["support"])
        return self

    def __invert__(self):
        self.parameters["support"] = operator.invert(self.parameters["support"])
        return self


class Normal(Distribution):
    """
    The (multi-variate) normal distribution.

    The Gaussian distribution is ubiquitous in probability theory, since it is the final and stable or equilibrium
    distribution to which other distributions gravitate under a wide variety of smooth operations, e.g.,
    convolutions and stochastic transformations. One example of this is the central limit theorem. The Gaussian
    distribution is also attractive from a numerical point of view as it is maintained through many transformations
    (e.g. it is stable).

    Parameters
    ----------
    mean : array-like or LinearOperator
        Mean of the normal distribution.

    cov : array-like or LinearOperator
        (Co-)variance of the normal distribution.

    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to use for drawing
        realizations from this distribution.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    Distribution : Class representing general probability distributions.

    Examples
    --------
    >>> from probnum.probability import Normal
    >>> N = Normal(mean=0.5, cov=1)
    >>> N1 = 2*N - 1
    >>> N1.parameters
    {'mean': 0.0, 'cov': 4.0}
    """

    def __init__(self, mean=0, cov=1, random_state=None):
        # Set dtype to float
        _dtype = float

        # Check input for univariate, multivariate, matrix-variate or operator-variate
        if np.isscalar(mean) and np.isscalar(cov):
            self._normal_type = "scalar"
        elif isinstance(mean, (np.ndarray, scipy.sparse.spmatrix,)) and isinstance(cov,
                                                                                   (np.ndarray, scipy.sparse.spmatrix)):
            if len(mean.shape) == 1:
                self._normal_type = "vector"
            else:
                self._normal_type = "matrix"
        elif isinstance(mean, scipy.sparse.linalg.LinearOperator) or isinstance(cov,
                                                                                scipy.sparse.linalg.LinearOperator):
            self._normal_type = "operator"
        else:
            raise ValueError(
                "Cannot instantiate normal distribution with mean of type {} and covariance of type {}.".format(
                    mean.__class__.__name__, cov.__class__.__name__))

        # Check shape mismatch of mean and covariance
        _mean_dim = np.prod(mean.shape)
        if self._normal_type in ["scalar", "vector", "matrix"]:
            if _mean_dim != cov.shape[0] or _mean_dim != cov.shape[1]:
                raise ValueError(
                    "Shape mismatch of mean and covariance. Total number of elements of the mean must match " +
                    "the first and second dimension of the covariance.")
        elif self._normal_type == "operator":
            m, n = mean.shape
            # Kronecker structured covariance
            if isinstance(cov, linear_operators.Kronecker):
                # If mean has dimension (m x n) then covariance factors must be (m x m) and (n x n)
                if m != cov.A.shape[0] or m != cov.A.shape[1] or n != cov.B.shape[0] or n != cov.B.shape[1]:
                    raise ValueError(
                        "Kronecker structured covariance must have factors with the same shape as the mean.")
            # Symmetric Kronecker structured covariance
            if isinstance(cov, linear_operators.SymmetricKronecker):
                # Mean has to be square. If mean has dimension (n x n) then covariance factors must be (n x n).
                if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
                    raise ValueError(
                        "Normal distribution with symmetric Kronecker structured covariance must have square mean"
                        + " and square covariance factors with matching dimensions."
                    )
            # General case
            if _mean_dim != cov.shape[0] or _mean_dim != cov.shape[1]:
                raise ValueError(
                    "Shape mismatch of mean and covariance."
                )

        # Call to super class initiator
        super().__init__(parameters={"mean": mean, "cov": cov}, dtype=_dtype, random_state=random_state)

    # TODO: allow for linear operators as mean and covariance (pdf, logpdf computation, sampling, etc...)
    # TODO: implement (more efficient) versions of these functions (for linear operators)
    # TODO: refactor into superclass with subclasses (_ScalarNormal, _VectorNormal, _MatrixNormal, _LinOpNormal)
    def pdf(self, x):
        if self._normal_type == "scalar":
            return scipy.stats.norm.pdf(x, loc=self.parameters["mean"], scale=np.sqrt(self.parameters["cov"]))
        if self._normal_type == "vector":
            return scipy.stats.multivariate_normal.pdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def logpdf(self, x):
        if self._normal_type == "scalar":
            return scipy.stats.norm.logpdf(x, loc=self.parameters["mean"], scale=np.sqrt(self.parameters["cov"]))
        if self._normal_type == "vector":
            return scipy.stats.multivariate_normal.logpdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def cdf(self, x):
        if self._normal_type == "scalar":
            return scipy.stats.norm.cdf(x, loc=self.parameters["mean"], scale=np.sqrt(self.parameters["cov"]))
        if self._normal_type == "vector":
            return scipy.stats.multivariate_normal.cdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def logcdf(self, x):
        if self._normal_type == "scalar":
            return scipy.stats.norm.logcdf(x, loc=self.parameters["mean"], scale=np.sqrt(self.parameters["cov"]))
        if self._normal_type == "vector":
            return scipy.stats.multivariate_normal.logcdf(x, mean=self.parameters["mean"], cov=self.parameters["cov"])

    def sample(self, size=()):
        if self._normal_type == "scalar":
            return scipy.stats.norm.rvs(loc=self.parameters["mean"], scale=self.std(),
                                        size=size, random_state=self.random_state)
        if self._normal_type == "vector":
            return scipy.stats.multivariate_normal.rvs(mean=self.parameters["mean"], cov=self.parameters["cov"],
                                                       size=size, random_state=self.random_state)

    def mean(self):
        return self.parameters["mean"]

    def var(self):
        if self._normal_type == "scalar":
            return self.parameters["cov"]
        if self._normal_type in ["vector", "matrix"]:
            return np.diag(self.parameters["cov"])
        if self._normal_type == "operator":
            return linear_operators.Diagonal(Op=self.parameters["cov"])

    # def reshape(self, shape):
    #     try:
    #         # Reshape mean and covariance
    #         self._parameters["mean"].reshape(shape=shape)
    #         # self._parameters["cov"]. # TODO: how to realize this? Need to implement Matrix-variate normal first.
    #     except ValueError:
    #         raise ValueError("Cannot reshape this Normal distribution to the given shape: {}".format(str(shape)))

    # Binary arithmetic operations
    def __add__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            return Normal(mean=self.parameters["mean"] + delta,
                          cov=self.parameters["cov"],
                          random_state=self.random_state)
        else:
            raise NotImplementedError(
                "Addition not implemented for {} and {}.".format(self.__class__.__name__, other.__class__.__name__))

    def __sub__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return self + (-otherdist)
        else:
            raise NotImplementedError(
                "Subtraction not implemented for {} and {}.".format(self.__class__.__name__, other.__class__.__name__))

    def __mul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            if delta == 0:
                return Dirac(support=0 * self.mean(), random_state=self.random_state)
            else:
                return Normal(mean=self.parameters["mean"] * delta,
                              cov=self.parameters["cov"] * delta ** 2,
                              random_state=self.random_state)
        else:
            raise NotImplementedError(
                "Multiplication not implemented for {} and {}.".format(self.__class__.__name__,
                                                                       other.__class__.__name__))

    def __matmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            return Normal(mean=self.parameters["mean"] @ delta,
                          cov=delta @ (self.parameters["cov"] @ delta.transpose()),
                          random_state=self.random_state)
        raise NotImplementedError(
            "Matrix multiplication not implemented for {} and {}.".format(self.__class__.__name__,
                                                                          other.__class__.__name__))

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division by zero not supported.")
        else:
            otherdist = asdist(other)
            if isinstance(otherdist, Dirac):
                return self * operator.inv(otherdist)
            else:
                raise NotImplementedError(
                    "Division not implemented for {} and {}.".format(self.__class__.__name__, other.__class__.__name__))

    def __pow__(self, power, modulo=None):
        raise NotImplementedError(
            "Exponentiation not implemented for {} and {}.".format(self.__class__.__name__, power.__class__.__name__))

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return self + otherdist
        else:
            raise NotImplementedError(
                "Addition not implemented for {} and {}.".format(other.__class__.__name__, self.__class__.__name__))

    def __rsub__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return operator.neg(self) + otherdist
        else:
            raise NotImplementedError(
                "Subtraction not implemented for {} and {}.".format(other.__class__.__name__, self.__class__.__name__))

    def __rmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return self * otherdist
        else:
            raise NotImplementedError(
                "Multiplication not implemented for {} and {}.".format(other.__class__.__name__,
                                                                       self.__class__.__name__))

    def __rmatmul__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            delta = otherdist.mean()
            return Normal(mean=delta @ self.parameters["mean"],
                          cov=delta @ (self.parameters["cov"] @ delta.transpose()),
                          random_state=self.random_state)
        raise NotImplementedError(
            "Matrix multiplication not implemented for {} and {}.".format(other.__class__.__name__,
                                                                          self.__class__.__name__))

    def __rtruediv__(self, other):
        otherdist = asdist(other)
        if isinstance(otherdist, Dirac):
            return operator.inv(self) * otherdist
        else:
            raise NotImplementedError(
                "Division not implemented for {} and {}.".format(other.__class__.__name__, self.__class__.__name__))

    def __rpow__(self, power, modulo=None):
        raise NotImplementedError(
            "Exponentiation not implemented for {} and {}.".format(power.__class__.__name__, self.__class__.__name__))

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
        try:
            return Normal(mean=- self.parameters["mean"],
                          cov=self.parameters["cov"],
                          random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Negation not implemented for {}.".format(self.__class__.__name__))

    def __pos__(self):
        try:
            return Normal(mean=operator.pos(self.parameters["mean"]),
                          cov=self.parameters["cov"],
                          random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Negation not implemented for {}.".format(self.__class__.__name__))

    def __abs__(self):
        try:
            # todo: add absolute moments of normal (see: https://arxiv.org/pdf/1209.4340.pdf)
            return Distribution(parameters={},
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Absolute value not implemented for {}.".format(self.__class__.__name__))

    def __invert__(self):
        try:
            return Distribution(parameters={},
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            raise NotImplementedError(
                "Inversion not implemented for {}.".format(self.__class__.__name__))


def asrandvar(obj):
    """
    Return `obj` as a :class:`RandomVariable`.

    Converts scalars, (sparse) arrays or distribution classes to a :`class:RandomVariable`.

    Parameters
    ----------
    obj : object
        Argument to be represented as a :`class:RandomVariable`.

    Returns
    -------
    rv : RandomVariable
        The object `obj` as a as a :`class:RandomVariable`.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Examples
    --------
    >>> from probnum.probability import asrandvar
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> asrandvar(M)
    <2x3 RandomVariable with dtype=int32>
    """
    # RandomVariable
    if isinstance(obj, RandomVariable):
        return obj
    # Scalar
    elif np.isscalar(obj):
        return RandomVariable(shape=(), dtype=type(obj), distribution=Dirac(support=obj))
    # Numpy array, sparse array or Linear Operator
    elif isinstance(obj, (np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator)):
        return RandomVariable(shape=obj.shape, dtype=obj.dtype, distribution=Dirac(support=obj))
    elif isinstance(obj, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
        # TODO: transform scipy distribution objects (rvs?), Distribution objects automatically
        raise NotImplementedError
    else:
        raise ValueError("Argument of type {} cannot be converted to a random variable.".format(type(obj)))


def asdist(obj):
    """
    Return ``obj`` as a :class:`Distribution`.

    Converts scalars, (sparse) arrays or distribution classes to a :class:`Distribution`.

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
    Distribution : Class representing probability distributions.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> from probnum.probability import asdist
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
    # Scipy distribution
    elif isinstance(obj, scipy.stats._distn_infrastructure.rv_frozen):
        # Normal distribution
        if obj.dist.name == "norm":
            return Normal(mean=obj.mean(), cov=obj.var(), random_state=obj.random_state)
        elif obj.__class__.__name__ == "multivariate_normal_frozen":  # Multivariate normal
            return Normal(mean=obj.mean, cov=obj.cov, random_state=obj.random_state)
        else:
            # Generic distribution
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
    else:
        try:
            # Numpy array
            return Dirac(support=np.array(obj))
        except Exception:
            raise NotImplementedError("Cannot convert object of type {} to a distribution.".format(type(obj)))
