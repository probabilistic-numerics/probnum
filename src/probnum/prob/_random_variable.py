"""
Random Variables.

This module implements random variables. Random variables are the main in- and outputs
of probabilistic numerical methods.
"""

from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum import utils as _utils


ValueType = TypeVar("ValueType")
RandomStateType = Union[  # see scipy._lib._util.check_random_state
    None, int, np.random.RandomState, np.random.Generator
]


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class RandomVariable(Generic[ValueType]):
    """
    Random variables are the main objects used by probabilistic numerical methods.

    Every probabilistic numerical method takes a random variable encoding the prior
    distribution as input and outputs a random variable whose distribution encodes the
    uncertainty arising from finite computation. The generic signature of a
    probabilistic numerical method is:

    ``output_rv = probnum_method(input_rv, method_params)``

    In practice, most random variables used by methods in ProbNum have Dirac or Gaussian
    measure.

    Instances of :class:`RandomVariable` can be added, multiplied, etc. with arrays and
    linear operators. This may change their ``distribution`` and not necessarily all
    previously available methods are retained.

    Parameters
    ----------
    shape : tuple
        Shape of realizations of this random variable.
    dtype : numpy.dtype or object
        Data type of realizations of this random variable. If ``object`` will be
        converted to ``numpy.dtype``.
    distribution : Distribution
        Probability distribution of the random variable.

    See Also
    --------
    asrandvar : Transform into a :class:`RandomVariable`.
    Distribution : A class representing probability distributions.

    Examples
    --------
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: np.dtype,
        random_state: Optional[RandomStateType] = None,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[int], ValueType]] = None,
        in_support: Optional[Callable[[ValueType], bool]] = None,
        pdf: Optional[Callable[[ValueType], float]] = None,
        logpdf: Optional[Callable[[ValueType], float]] = None,
        cdf: Optional[Callable[[ValueType], float]] = None,
        logcdf: Optional[Callable[[ValueType], float]] = None,
        quantile: Optional[Callable[[float], ValueType]] = None,
        mode: Optional[Callable[[], ValueType]] = None,
        median: Optional[Callable[[], ValueType]] = None,
        mean: Optional[Callable[[], ValueType]] = None,
        cov: Optional[Callable[[], ValueType]] = None,
        var: Optional[Callable[[], ValueType]] = None,
        std: Optional[Callable[[], ValueType]] = None,
        entropy: Optional[Callable[[], float]] = None,
    ):
        """Create a new random variable."""
        self._shape = RandomVariable._check_shape(shape)
        self._dtype = dtype

        self._random_state = scipy._lib._util.check_random_state(random_state)

        # Probability distribution of the random variable
        self._parameters = parameters.copy() if parameters is not None else {}

        self.__sample = sample

        self.__in_support = in_support
        self.__pdf = pdf
        self.__logpdf = logpdf
        self.__cdf = cdf
        self.__logcdf = logcdf
        self.__quantile = quantile

        # Properties of the random variable
        self.__mode = mode
        self.__median = median
        self.__mean = mean
        self.__cov = cov
        self.__var = var
        self.__std = std
        self.__entropy = entropy

    @staticmethod
    def _check_shape(
        shape: Optional[Union[int, Tuple[int, ...]]]
    ) -> Optional[Tuple[int, ...]]:
        if shape is None:
            return None
        elif isinstance(shape, tuple) and all(
            isinstance(entry, int) for entry in shape
        ):
            return shape
        elif isinstance(shape, int):
            return (shape,)
        else:
            raise TypeError(
                f"The given shape {shape} is not an int or a tuple of ints."
            )

    def __repr__(self) -> str:
        return f"<{self.shape} {self.__class__.__name__} with dtype={self.dtype}>"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of realizations of the random variable."""
        return self._shape

    @cached_property
    def ndim(self) -> int:
        return len(self._shape)

    @cached_property
    def size(self) -> int:
        return int(np.prod(self._shape))

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) a realization of this random variable."""
        return self._dtype

    @property
    def random_state(self) -> Union[np.random.RandomState, np.random.Generator]:
        """Random state of the random variable.

        This attribute defines the RandomState object to use for drawing
        realizations from this random variable.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState`
        instance.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed: RandomStateType):
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

        The parameters of the distribution such as mean, variance, et cetera stored in a
        ``dict``.
        """
        return self._parameters.copy()

    @staticmethod
    def _check_property_value(
        name: str,
        value: Any,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
    ):
        if shape is not None:
            if value.shape != shape:
                raise ValueError(
                    f"The {name} of the random variable does not have the correct "
                    f"shape. Expected {shape} but got {value.shape}."
                )

        if dtype is not None:
            if not np.issubdtype(value.dtype, dtype):
                raise ValueError(
                    f"The {name} of the random variable does not have the correct "
                    f"dtype. Expected {dtype.name} but got {value.dtype.name}."
                )

    @cached_property
    def mode(self):
        """
        Mode of the random variable.

        Returns
        -------
        mode : float
            The mode of the random variable.
        """
        if self.__mode is None:
            raise NotImplementedError

        mode = self.__mode()

        RandomVariable._check_property_value(
            "mode", mode, shape=self._shape, dtype=self._dtype,
        )

        return mode

    @cached_property
    def median(self):
        """
        Median of the random variable.

        Returns
        -------
        median : float
            The median of the distribution.
        """

        if self._shape != ():
            raise NotImplementedError(
                "The median is only defined for scalar random variables."
            )

        if self.__median is None:
            try:
                median = self.quantile(0.5)
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            median = self.__median()

        RandomVariable._check_property_value(
            "median", median, shape=(), dtype=self._dtype,
        )

        return median

    @cached_property
    def mean(self):
        """
        Mean :math:`\\mathbb{E}(X)` of the distribution.

        Returns
        -------
        mean : array-like
            The mean of the distribution.
        """
        if self.__mean is None:
            raise NotImplementedError

        mean = self.__mean()

        RandomVariable._check_property_value(
            "mean", mean, shape=self._shape, dtype=self._dtype,
        )

        return mean

    @cached_property
    def cov(self):
        """
        Covariance :math:`\\operatorname{Cov}(X) = \\mathbb{E}((X-\\mathbb{E}(X))(X-\\mathbb{E}(X))^\\top)`
        of the random variable.

        Returns
        -------
        cov : array-like
            The kernels of the random variable.
        """  # pylint: disable=line-too-long
        if self.__cov is None:
            raise NotImplementedError

        cov = self.__cov()

        RandomVariable._check_property_value(
            "covariance",
            cov,
            shape=(self.size, self.size) if self.ndim > 0 else (),
            dtype=self._dtype,
        )

        return cov

    @cached_property
    def var(self):
        """
        Variance :math:`\\operatorname{Var}(X) = \\mathbb{E}((X-\\mathbb{E}(X))^2)` of
        the distribution.

        Returns
        -------
        var : array-like
            The variance of the distribution.
        """
        if self.__var is None:
            try:
                var = np.diag(self.cov).reshape(self._shape).copy()
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            var = self.__var()

        RandomVariable._check_property_value(
            "variance", var, shape=self._shape, dtype=self._dtype,
        )

        return var

    @cached_property
    def std(self):
        """
        Standard deviation of the distribution.

        Returns
        -------
        std : array-like
            The standard deviation of the distribution.
        """
        if self.__std is None:
            try:
                std = np.sqrt(self.var)
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            std = self.__std()

        RandomVariable._check_property_value(
            "standard deviation", std, shape=self._shape, dtype=self._dtype,
        )

        return std

    @cached_property
    def entropy(self):
        if self.__entropy is None:
            raise NotImplementedError

        entropy = self.__entropy()

        RandomVariable._check_property_value(
            "entropy", entropy, shape=(), dtype=np.floating,
        )

        return entropy

    def in_support(self, x: ValueType) -> bool:
        if self.__in_support is None:
            raise NotImplementedError

        return self.__in_support(x)

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
        if self.__sample is None:
            raise NotImplementedError("No sampling method provided.")

        return self.__sample(size=size)

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
        if self.__pdf is not None:
            return self.__pdf(x)
        if self.__logpdf is not None:
            return np.exp(self.__logpdf(x))
        raise NotImplementedError(
            "The function 'pdf' is not implemented for object of class {}".format(
                type(self).__name__
            )
        )

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
        if self.__logpdf is not None:
            return self.__logpdf(x)
        elif self.__pdf is not None:
            return np.log(self.__pdf(x))
        else:
            raise NotImplementedError(
                f"The function 'logpdf' is not implemented for object of class "
                f"{type(self).__name__}."
            )

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
        if self.__cdf is not None:
            return self.__cdf(x)
        elif self.__logcdf is not None:
            return np.exp(self.__logcdf(x))
        else:
            raise NotImplementedError(
                "The function 'cdf' is not implemented for object of class {}".format(
                    type(self).__name__
                )
            )

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
        if self.__logcdf is not None:
            return self.__logcdf(x)
        elif self.__cdf is not None:
            return np.log(self.__cdf(x))
        else:
            raise NotImplementedError(
                f"The function 'logcdf' is not implemented for object of class "
                f"{type(self).__name__}."
            )

    def quantile(self, p: Union[float, np.floating]) -> ValueType:
        if self.__quantile is None:
            raise NotImplementedError

        return self.__quantile(p)

    def reshape(self, newshape):
        """
        Give a new shape to a random variable.

        Parameters
        ----------
        newshape : int or tuple of ints
            New shape for the random variable. It must be compatible with the original
            shape.

        Returns
        -------
        reshaped_rv : ``self`` with the new dimensions of ``shape``.
        """
        raise NotImplementedError(
            f"Reshaping not implemented for random variables of type: "
            f"{self.__class__.__name__}."
        )

    def transpose(self, *axes):
        """
        Transpose the random variable.

        Parameters
        ----------
        axes : None, tuple of ints, or n ints
            See documentation of numpy.ndarray.transpose.

        Returns
        -------
        transposed_rv : The transposed random variable.
        """
        raise NotImplementedError(
            f"Transposition not implemented for random variables of type: "
            f"{self.__class__.__name__}."
        )

    T = property(transpose)

    # Unary arithmetic operations

    def __neg__(self) -> "RandomVariable":
        return NotImplemented

    def __pos__(self) -> "RandomVariable":
        return NotImplemented

    def __abs__(self) -> "RandomVariable":
        return RandomVariable(
            shape=self.shape,
            dtype=self.dtype,
            random_state=_utils.derive_random_seed(self.random_state),
            sample=lambda size: abs(self.sample(size=size)),
        )

    # Binary arithmetic operations

    __array_ufunc__ = None
    """
    This prevents numpy from calling elementwise arithmetic
    operations allowing expressions like: y = np.array([1, 1]) + RV
    to call the arithmetic operations defined by RandomVariable
    instead of elementwise. Thus no array of RandomVariables but a
    RandomVariable with the correct shape is returned.
    """

    def __add__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self + asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import add

        return add(self, other)

    def __radd__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) + self

        return NotImplemented

    def __sub__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self - asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import sub

        return sub(self, other)

    def __rsub__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) - self

        return NotImplemented

    def __mul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self * asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import mul

        return mul(self, other)

    def __rmul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) * self

        return NotImplemented

    def __matmul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self @ asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import matmul

        return matmul(self, other)

    def __rmatmul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) @ self

        return NotImplemented

    def __truediv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self / asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import truediv

        return truediv(self, other)

    def __rtruediv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) / self

        return NotImplemented

    def __floordiv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self // asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import floordiv

        return floordiv(self, other)

    def __rfloordiv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) // self

        return NotImplemented

    def __mod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self % asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import mod

        return mod(self, other)

    def __rmod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) % self

        return NotImplemented

    def __divmod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return divmod(self, asrandvar(other))

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import divmod_

        return divmod_(self, other)

    def __rdivmod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return divmod(asrandvar(other), self)

        return NotImplemented

    def __pow__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self ** asrandvar(other)

        # pylint: disable=import-outside-toplevel
        from .random_variable._arithmetic import pow_

        return pow_(self, other)

    def __rpow__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) ** self

        return NotImplemented


def asrandvar(obj) -> RandomVariable:
    """
    Return ``obj`` as a :class:`RandomVariable`.

    Converts scalars, (sparse) arrays or distribution classes to a
    :class:`RandomVariable`.

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
    array([1, 1, 1, 0, 0])
    """

    # pylint: disable=import-outside-toplevel
    from probnum.prob.random_variable import Dirac

    # RandomVariable
    if isinstance(obj, RandomVariable):
        return obj
    # Scalar
    elif np.isscalar(obj):
        return Dirac(support=obj)
    # Numpy array, sparse array or Linear Operator
    elif isinstance(
        obj, (np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator)
    ):
        return Dirac(support=obj)
    # Scipy random variable
    elif isinstance(
        obj,
        (
            scipy.stats._distn_infrastructure.rv_frozen,
            scipy.stats._multivariate.multi_rv_frozen,
        ),
    ):
        return _scipystats_to_rv(scipyrv=obj)
    else:
        raise ValueError(
            f"Argument of type {type(obj)} cannot be converted to a random variable."
        )


def _scipystats_to_rv(
    scipyrv: Union[
        scipy.stats._distn_infrastructure.rv_frozen,
        scipy.stats._multivariate.multi_rv_frozen,
    ]
):
    """
    Transform SciPy distributions to Probnum :class:`RandomVariable`s.

    Parameters
    ----------
    scipyrv :
        SciPy distribution.

    Returns
    -------
    dist : RandomVariable
        ProbNum random variable.

    """

    # pylint: disable=import-outside-toplevel
    from probnum.prob.random_variable import Normal

    # Univariate distributions (implemented in this package)
    if isinstance(scipyrv, scipy.stats._distn_infrastructure.rv_frozen):
        # Normal distribution
        if scipyrv.dist.name == "norm":
            return Normal(
                mean=scipyrv.mean(),
                cov=scipyrv.var(),
                random_state=scipyrv.random_state,
            )
    # Multivariate distributions (implemented in this package)
    elif isinstance(scipyrv, scipy.stats._multivariate.multi_rv_frozen):
        # Multivariate normal
        if scipyrv.__class__.__name__ == "multivariate_normal_frozen":
            return Normal(
                mean=scipyrv.mean, cov=scipyrv.cov, random_state=scipyrv.random_state,
            )
    # Generic distributions
    if (
        hasattr(scipyrv, "dist") and isinstance(scipyrv.dist, scipy.stats.rv_discrete)
    ) or hasattr(scipyrv, "pmf"):
        pdf = getattr(scipyrv, "pmf", None)
        logpdf = getattr(scipyrv, "logpmf", None)
    else:
        pdf = getattr(scipyrv, "pdf", None)
        logpdf = getattr(scipyrv, "logpdf", None)

    def _wrap_np_scalar(fn):
        if fn is None:
            return None

        def _wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)

            if np.isscalar(res):
                return _utils.as_numpy_scalar(res)

            return res

        return _wrapper

    # Infer shape and dtype
    sample = _wrap_np_scalar(scipyrv.rvs)()

    return RandomVariable(
        shape=sample.shape,
        dtype=sample.dtype,
        random_state=getattr(scipyrv, "random_state", None),
        sample=_wrap_np_scalar(getattr(scipyrv, "rvs", None)),
        in_support=None,  # TODO for univariate
        pdf=_wrap_np_scalar(pdf),
        logpdf=_wrap_np_scalar(logpdf),
        cdf=_wrap_np_scalar(getattr(scipyrv, "cdf", None)),
        logcdf=_wrap_np_scalar(getattr(scipyrv, "logcdf", None)),
        quantile=_wrap_np_scalar(getattr(scipyrv, "ppf", None)),
        mode=None,  # not offered by scipy.stats
        median=_wrap_np_scalar(getattr(scipyrv, "median", None)),
        mean=_wrap_np_scalar(getattr(scipyrv, "mean", None)),
        cov=_wrap_np_scalar(getattr(scipyrv, "cov", None)),
        var=_wrap_np_scalar(getattr(scipyrv, "var", None)),
        std=_wrap_np_scalar(getattr(scipyrv, "std", None)),
        entropy=_wrap_np_scalar(getattr(scipyrv, "entropy", None)),
    )
