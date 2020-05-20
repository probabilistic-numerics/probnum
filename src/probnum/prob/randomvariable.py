"""
Random Variables.

This module implements random variables. Random variables are the main in- and outputs of probabilistic numerical
methods.
"""

import operator

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.prob.distributions.distribution import Distribution
from probnum.prob.distributions.dirac import Dirac
from probnum.prob.distributions.normal import Normal


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
    asrandvar : Transform into a :class:`RandomVariable`.
    Distribution : A class representing probability distributions.

    Examples
    --------
    """

    def __init__(self, shape=None, dtype=None, distribution=None):
        """Create a new random variable."""
        self._set_distribution(distribution)
        self._set_dtype(distribution, dtype)
        self._set_shape(distribution, shape)

    def _set_distribution(self, distribution):
        """
        Set distribution of random variable.
        """
        if isinstance(distribution, Distribution):
            self._distribution = distribution
        elif distribution is None:
            self._distribution = Distribution()
        else:
            raise ValueError("The distribution parameter must be an "
                             "instance of `Distribution`.")

    # TODO: add some type checking (e.g. for shape as a tuple of ints) and extract as function
    def _set_shape(self, distribution, shape):
        """
        Sets shape in accordance with distribution.
        """
        self._shape = shape
        if distribution is not None:
            if distribution.shape is not None:
                if shape is None or distribution.shape == shape:
                    self._shape = distribution.shape
                else:
                    raise ValueError("Shape of distribution and given shape do not match.")
            else:
                self.distribution.reshape(newshape=shape)

    def _set_dtype(self, distribution, dtype):
        """
        Sets dtype in accordance with distribution.
        """
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
        if self._dtype is None:
            raise ValueError("No \'dtype\' set.")

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
                return self._distribution.cov()
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
        """Data type of (elements of) a realization of this random variable."""
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

    def reshape(self, newshape):
        """
        Give a new shape to a random variable.

        Parameters
        ----------
        newshape : int or tuple of ints
            New shape for the random variable. It must be compatible with the original shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.
        """
        self._shape = newshape
        self._distribution.reshape(newshape=newshape)
        return self

    # Binary arithmetic operations

    # The below prevents numpy from calling elementwise arithmetic
    # operations allowing expressions like: y = np.array([1, 1]) + RV
    # to call the arithmetic operations defined by RandomVariable
    # instead of elementwise. Thus no array of RandomVariables but a
    # RandomVariable with the correct shape is returned.
    __array_ufunc__ = None

    def _rv_from_binary_operation(self, other, op):
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
        return self._rv_from_binary_operation(other=other, op=operator.add)

    def __sub__(self, other):
        return self._rv_from_binary_operation(other=other, op=operator.sub)

    def __mul__(self, other):
        return self._rv_from_binary_operation(other=other, op=operator.mul)

    def __matmul__(self, other):
        return self._rv_from_binary_operation(other=other, op=operator.matmul)

    def __truediv__(self, other):
        return self._rv_from_binary_operation(other=other, op=operator.truediv)

    def __pow__(self, power, modulo=None):
        return self._rv_from_binary_operation(other=power, op=operator.pow)

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_binary_operation(other=self, op=operator.add)

    def __rsub__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_binary_operation(other=self, op=operator.sub)

    def __rmul__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_binary_operation(other=self, op=operator.mul)

    def __rmatmul__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_binary_operation(other=self, op=operator.matmul)

    def __rtruediv__(self, other):
        other_rv = asrandvar(other)
        return other_rv._rv_from_binary_operation(other=self, op=operator.truediv)

    def __rpow__(self, power, modulo=None):
        other_rv = asrandvar(power)
        return other_rv._rv_from_binary_operation(other=self, op=operator.pow)

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    # TODO: needs setter functions for properties `shape` and `dtype` to do in place
    def __iadd__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __imatmul__(self, other):
        return NotImplemented

    def __itruediv__(self, other):
        return NotImplemented

    def __ipow__(self, power, modulo=None):
        return NotImplemented

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


def asrandvar(obj):
    """
    Return ``obj`` as a :class:`RandomVariable`.

    Converts scalars, (sparse) arrays or distribution classes to a :class:`RandomVariable`.

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
    >>> from probnum.prob import asrandvar
    >>> bern = bernoulli(p=0.5)
    >>> bern.random_state = 42  # Seed for reproducibility
    >>> b = asrandvar(bern)
    >>> b.sample(size=5)
    array([0, 1, 1, 1, 0])
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
    # Scipy random variable
    # elif isinstance(obj, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
    elif isinstance(obj, scipy.stats._distn_infrastructure.rv_frozen):
        return _scipystats_to_rv(obj=obj)
    else:
        raise ValueError("Argument of type {} cannot be converted to a random variable.".format(type(obj)))


def _scipystats_to_rv(obj):
    """
    Transform SciPy distributions to Probnum :class:`RandomVariable`s.

    Parameters
    ----------
    obj : object
        SciPy distribution.

    Returns
    -------
    dist : RandomVariable
        ProbNum random variable.

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
                            cov=obj.var,
                            random_state=obj.random_state)
