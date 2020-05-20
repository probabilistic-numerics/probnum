"""
Probability Distribution.

This module provides a class implementing a probability distribution along with its properties.
"""

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util


class Distribution:
    """
    A class representing probability distributions.

    This class is primarily intended to be subclassed to provide
    distribution-specific implementations of the various methods
    (``logpdf``, ``logcdf``, ``sample``, ``mean``, ``var``, ...). When
    creating a subclass implementing a certain distribution for which
    operations like addition or multiplication are available, please
    consider implementing them, as done for instance in :class:`Normal`.
    The overriden methods should also update the ``parameters`` of the
    distribution. This allows arithmetic operations on instances of
    :class:`Distribution` and :class:`RandomVariable`.

    Parameters
    ----------
    parameters : dict
        Dictionary of distribution parameters such as mean, variance, shape, etc.
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
    cov : callable
        Function returning the covariance of the distribution.
    shape : tuple
        Shape of samples from this distribution.
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
    RandomVariable : Random variables are the main objects used by probabilistic numerical methods.

    Examples
    --------

    """

    def __init__(self, parameters=None, pdf=None, logpdf=None, cdf=None, logcdf=None, sample=None,
                 mean=None, cov=None, shape=None, dtype=None, random_state=None):
        if parameters is None:
            parameters = {}  # sentinel value to avoid anti-pattern
        self._parameters = parameters
        self._pdf = pdf
        self._logpdf = logpdf
        self._cdf = cdf
        self._logcdf = logcdf
        self._sample = sample
        self._mean = mean
        self._cov = cov
        self._set_shape(shape)
        self._dtype = dtype
        self._random_state = scipy._lib._util.check_random_state(random_state)

    def _check_distparams(self):
        pass
        # TODO: type checking of self._parameters; check method signatures.

    def _set_shape(self, shape):
        """
        Sets shape in accordance with distribution mean.
        """
        self._shape = shape
        try:
            # Set shape based on mean
            if np.isscalar(self.mean()):
                shape_mean = ()
            else:
                shape_mean = self.mean().shape
            if shape is None or shape_mean == shape:
                self._shape = shape_mean
            else:
                raise ValueError("Shape of distribution mean and given shape do not match.")
        except NotImplementedError:
            # Set shape based on a sample
            if np.isscalar(self.sample(size=1)):
                shape_sample = ()
            else:
                shape_sample = self.sample(size=1).shape
            if shape is None or shape_sample == shape:
                self._shape = shape_sample
            else:
                raise ValueError("Shape of distribution mean and given shape do not match.")

    @property
    def shape(self):
        """Shape of samples from this distribution."""
        return self._shape

    @property
    def dtype(self):
        """``Dtype`` of elements of samples from this distribution."""
        return self._dtype

    @dtype.setter
    def dtype(self, newtype):
        """Set the ``dtype`` of the distribution."""
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
            raise AttributeError("No parameters of {} are available.".format(type(self).__name__))

    def pdf(self, x):
        """
        Probability density or mass function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the probability density / mass function.

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
        if self._logpdf is not None:
            return self._logpdf(x)
        elif self._pdf is not None:
            return np.log(self._pdf(x))
        else:
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
        elif self._logcdf is not None:
            return np.exp(self._logcdf(x))
        else:
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
        elif self._cdf is not None:
            return np.log(self._cdf(x))
        else:
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
        else:
            raise NotImplementedError(
                'The function \'sample\' is not implemented for object of class {}.'.format(type(self).__name__))

    def median(self):
        """
        Median of the distribution.

        Returns
        -------
        median : float
            The median of the distribution.
        """
        return self.cdf(x=0.5)

    def mode(self):
        """
        Mode of the distribution.

        Returns
        -------
        mode : float
            The mode of the distribution.
        """
        if "mode" in self._parameters:
            return self._parameters["mode"]
        else:
            raise NotImplementedError(
                'The function \'mode\' is not implemented for object of class {}.'.format(type(self).__name__))

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
                'The function \'mean\' is not implemented for object of class {}.'.format(type(self).__name__))

    def cov(self):
        """
        Covariance :math:`\\operatorname{Cov}(X) = \\mathbb{E}((X-\\mathbb{E}(X))(X-\\mathbb{E}(X))^\\top)` of the
        distribution.

        Returns
        -------
        cov : array-like
            The covariance of the distribution.
        """
        if self._cov is not None:
            return self._cov()
        elif "cov" in self._parameters:
            return self._parameters["cov"]
        else:
            raise NotImplementedError(
                'The function \'cov\' is not implemented for object of class {}'.format(type(self).__name__))

    def var(self):
        """
        Variance :math:`\\operatorname{Var}(X) = \\mathbb{E}((X-\\mathbb{E}(X))^2)` of the distribution.

        Returns
        -------
        var : array-like
            The variance of the distribution.
        """
        if "var" in self._parameters:
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

    def reshape(self, newshape):
        """
        Give a new shape to (realizations of) this distribution.

        Parameters
        ----------
        newshape : int or tuple of ints
            New shape for the realizations and parameters of this distribution. It must be compatible with the original
            shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.
        """
        raise NotImplementedError(
            "Reshaping not implemented for distribution of type: {}.".format(self.__class__.__name__))
