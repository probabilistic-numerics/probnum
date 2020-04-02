"""
Probability Distributions.

This module provides a class implementing a probability distributions along with its properties.
"""

import operator

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.probability.distributions.normal import Normal

__all__ = ["Distribution", "asdist"]


class Distribution:
    """
    A class representing probability distributions.

    This class is primarily intended to be subclassed to provide distributions-specific implementations of the various
    methods (``logpdf``, ``logcdf``, ``sample``, ``mean``, ``var``, ...). When creating a subclass implementing a
    certain distributions, make sure to override generic operations (addition, multiplication, ...) to represent the
    properties of the distributions. The overriden methods should also update the ``parameters`` of the distributions.
    This allows arithmetic operations on instances of :class:`Distribution` and :class:`RandomVariable`.

    Parameters
    ----------
    parameters : dict
        Dictionary of distributions parameters such as mean, variance, et cetera.
    pdf : callable
        Probability density or mass function.
    logpdf : callable
        Log-probability density or mass function.
    cdf : callable
        Cumulative distributions function.
    logcdf : callable
        Log-cumulative distributions function.
    sample : callable
        Function implementing sampling. Must have signature ``sample(size=())``.
    mean : callable
        Function returning the mean of the distributions.
    var : callable
        Function returning the variance of the distributions.
    dtype : numpy.dtype or object
        Data type of realizations of a random variable with this distributions. If ``object`` will be converted to ``numpy.dtype``.
    random_state : None or int or :class:`~numpy.random.RandomState` instance, optional
        This parameter defines the RandomState object to use for drawing
        realizations from this distributions.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState` instance.
        Default is None.

    See Also
    --------
    asdist : Transform object into a :class:`Distribution`.
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
        """`Dtype` of elements of samples from this distributions."""
        return self._dtype

    @dtype.setter
    def dtype(self, newtype):
        """Set the `dtype` of the distributions."""
        self._dtype = newtype

    @property
    def random_state(self):
        """Random state of the distributions.

        This attribute defines the RandomState object to use for drawing
        realizations from this distributions.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState` instance.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        """ Get or set the RandomState object of the underlying distributions.

        This can be either None or an existing RandomState object.
        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.
        """
        self._random_state = scipy._lib._util.check_random_state(seed)

    @property
    def parameters(self):
        """
        Parameters of the probability distributions.

        The parameters of the distributions such as mean, variance, et cetera stored in a ``dict``.
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
        Cumulative distributions function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the cumulative distributions function.

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
        Log-cumulative distributions function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the cumulative distributions function.

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
        Median of the distributions.

        Returns
        -------
        median : float
            The median of the distributions.
        """
        return self.cdf(x=0.5)

    def mean(self):
        """
        Mean :math:`\\mathbb{E}(X)` of the distributions.

        Returns
        -------
        mean : array-like
            The mean of the distributions.
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
        Variance :math:`\\operatorname{Var}(X) = \\mathbb{E}((X-\\mathbb{E}(X))^2)` of the distributions.

        Returns
        -------
        var : array-like
            The variance of the distributions.
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
        Standard deviation of the distributions.

        Returns
        -------
        std : array-like
            The standard deviation of the distributions.
        """
        return np.sqrt(self.var())

    def reshape(self, shape):
        """
        Give a new shape to (realizations of) this distributions.

        Parameters
        ----------
        shape : int or tuple of ints
            New shape for the realizations and parameters of this distributions. It must be compatible with the original
            shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.
        """
        raise NotImplementedError(
            "Reshaping not implemented for distributions of type: {}.".format(self.__class__.__name__))

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
    # Scipy distributions
    elif isinstance(obj, scipy.stats._distn_infrastructure.rv_frozen):
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
    else:
        try:
            # Numpy array
            return Dirac(support=np.array(obj))
        except Exception:
            raise NotImplementedError("Cannot convert object of type {} to a distributions.".format(type(obj)))


class Dirac(Distribution):
    """
    The Dirac delta distributions.

    This distributions models a point mass and can be useful to represent numbers as random variables with Dirac measure.
    It has the useful property that arithmetic operations between a :class:`Dirac` random variable and an arbitrary
    :class:`RandomVariable` acts in the same way as the arithmetic operation with a constant.

    Note, that a Dirac measure does not admit a probability density function but can be viewed as a distributions
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
    >>> rv = RandomVariable(distributions=dist1 + dist2)
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
            raise ValueError("Cannot reshape this Dirac distributions to the given shape: {}".format(str(shape)))

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
