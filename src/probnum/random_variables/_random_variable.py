"""
Random Variables.

This module implements random variables. Random variables are the main in- and outputs
of probabilistic numerical methods.
"""

from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse
import scipy.stats

from probnum import utils as _utils
from probnum._lib.argtypes import (
    DTypeArgType,
    FloatArgType,
    RandomStateArgType,
    ShapeArgType,
)
from probnum.typing import RandomStateType, ShapeType

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


_ValueType = TypeVar("ValueType")


class RandomVariable(Generic[_ValueType]):
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

    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType,
        random_state: RandomStateArgType = None,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[ShapeType], _ValueType]] = None,
        in_support: Optional[Callable[[_ValueType], bool]] = None,
        cdf: Optional[Callable[[_ValueType], np.float_]] = None,
        logcdf: Optional[Callable[[_ValueType], np.float_]] = None,
        quantile: Optional[Callable[[FloatArgType], _ValueType]] = None,
        mode: Optional[Callable[[], _ValueType]] = None,
        median: Optional[Callable[[], _ValueType]] = None,
        mean: Optional[Callable[[], _ValueType]] = None,
        cov: Optional[Callable[[], _ValueType]] = None,
        var: Optional[Callable[[], _ValueType]] = None,
        std: Optional[Callable[[], _ValueType]] = None,
        entropy: Optional[Callable[[], np.float_]] = None,
    ):
        # pylint: disable=too-many-arguments,too-many-locals
        """Create a new random variable."""
        self._shape = _utils.as_shape(shape)
        self._dtype = np.dtype(dtype)

        self._random_state = _utils.as_random_state(random_state)

        # Probability distribution of the random variable
        self._parameters = parameters.copy() if parameters is not None else {}

        self.__sample = sample

        self.__in_support = in_support
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

    def __repr__(self) -> str:
        return f"<{self.shape} {self.__class__.__name__} with dtype={self.dtype}>"

    @property
    def shape(self) -> ShapeType:
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
    def random_state(self) -> RandomStateType:
        """Random state of the random variable.

        This attribute defines the RandomState object to use for drawing
        realizations from this random variable.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState`
        instance.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed: RandomStateArgType):
        """Get or set the RandomState object of the underlying distribution.

        This can be either None or an existing RandomState object.
        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.
        """
        self._random_state = _utils.as_random_state(seed)

    @property
    def parameters(self) -> Dict[str, Any]:
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
    def mode(self) -> _ValueType:
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
            "mode",
            mode,
            shape=self._shape,
            dtype=self._dtype,
        )

        # Make immutable
        if isinstance(mode, np.ndarray):
            mode.setflags(write=False)

        return mode

    @cached_property
    def median(self) -> _ValueType:
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
            "median",
            median,
            shape=(),
            dtype=self._dtype,
        )

        # Make immutable
        if isinstance(median, np.ndarray):
            median.setflags(write=False)

        return median

    @cached_property
    def mean(self) -> _ValueType:
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
            "mean",
            mean,
            shape=self._shape,
            dtype=self._dtype,
        )

        # Make immutable
        if isinstance(mean, np.ndarray):
            mean.setflags(write=False)

        return mean

    @cached_property
    def cov(self) -> _ValueType:
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

        # Make immutable
        if isinstance(cov, np.ndarray):
            cov.setflags(write=False)

        return cov

    @cached_property
    def var(self) -> _ValueType:
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
            "variance",
            var,
            shape=self._shape,
            dtype=self._dtype,
        )

        # Make immutable
        if isinstance(var, np.ndarray):
            var.setflags(write=False)

        return var

    @cached_property
    def std(self) -> _ValueType:
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
            "standard deviation",
            std,
            shape=self._shape,
            dtype=self._dtype,
        )

        # Make immutable
        if isinstance(std, np.ndarray):
            std.setflags(write=False)

        return std

    @cached_property
    def entropy(self) -> np.float_:
        if self.__entropy is None:
            raise NotImplementedError

        entropy = self.__entropy()

        RandomVariable._check_property_value(
            "entropy",
            entropy,
            shape=(),
            dtype=np.floating,
        )

        return entropy

    def in_support(self, x: _ValueType) -> bool:
        if self.__in_support is None:
            raise NotImplementedError

        return self.__in_support(x)

    def sample(self, size: ShapeArgType = ()) -> _ValueType:
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

        return self.__sample(size=_utils.as_shape(size))

    def cdf(self, x: _ValueType) -> np.float_:
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

    def logcdf(self, x: _ValueType) -> np.float_:
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

    def quantile(self, p: FloatArgType) -> _ValueType:
        if self.__quantile is None:
            raise NotImplementedError

        return self.__quantile(p)

    def reshape(self, newshape: ShapeArgType) -> "RandomVariable":
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

    def transpose(self, *axes: int) -> "RandomVariable":
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

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import add

        return add(self, other)

    def __radd__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) + self

        return NotImplemented

    def __sub__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self - asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import sub

        return sub(self, other)

    def __rsub__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) - self

        return NotImplemented

    def __mul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self * asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mul

        return mul(self, other)

    def __rmul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) * self

        return NotImplemented

    def __matmul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self @ asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import matmul

        return matmul(self, other)

    def __rmatmul__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) @ self

        return NotImplemented

    def __truediv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self / asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import truediv

        return truediv(self, other)

    def __rtruediv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) / self

        return NotImplemented

    def __floordiv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self // asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import floordiv

        return floordiv(self, other)

    def __rfloordiv__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) // self

        return NotImplemented

    def __mod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self % asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mod

        return mod(self, other)

    def __rmod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) % self

        return NotImplemented

    def __divmod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return divmod(self, asrandvar(other))

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import divmod_

        return divmod_(self, other)

    def __rdivmod__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return divmod(asrandvar(other), self)

        return NotImplemented

    def __pow__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return self ** asrandvar(other)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import pow_

        return pow_(self, other)

    def __rpow__(self, other) -> "RandomVariable":
        if not isinstance(other, RandomVariable):
            return asrandvar(other) ** self

        return NotImplemented


class DiscreteRandomVariable(RandomVariable[_ValueType]):
    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType,
        random_state: Optional[RandomStateType] = None,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[ShapeArgType], _ValueType]] = None,
        in_support: Optional[Callable[[_ValueType], bool]] = None,
        pmf: Optional[Callable[[_ValueType], np.float_]] = None,
        logpmf: Optional[Callable[[_ValueType], np.float_]] = None,
        cdf: Optional[Callable[[_ValueType], np.float_]] = None,
        logcdf: Optional[Callable[[_ValueType], np.float_]] = None,
        quantile: Optional[Callable[[FloatArgType], _ValueType]] = None,
        mode: Optional[Callable[[], _ValueType]] = None,
        median: Optional[Callable[[], _ValueType]] = None,
        mean: Optional[Callable[[], _ValueType]] = None,
        cov: Optional[Callable[[], _ValueType]] = None,
        var: Optional[Callable[[], _ValueType]] = None,
        std: Optional[Callable[[], _ValueType]] = None,
        entropy: Optional[Callable[[], np.float_]] = None,
    ):
        # Probability mass function
        self.__pmf = pmf
        self.__logpmf = logpmf

        super().__init__(
            shape=shape,
            dtype=dtype,
            random_state=random_state,
            parameters=parameters,
            sample=sample,
            in_support=in_support,
            cdf=cdf,
            logcdf=logcdf,
            quantile=quantile,
            mode=mode,
            median=median,
            mean=mean,
            cov=cov,
            var=var,
            std=std,
            entropy=entropy,
        )

    def pmf(self, x: _ValueType) -> np.float_:
        if self.__pmf is not None:
            return self.__pmf(x)
        elif self.__logpmf is not None:
            return np.exp(self.__logpmf(x))
        else:
            raise NotImplementedError

    def logpmf(self, x: _ValueType) -> np.float_:
        if self.__logpmf is not None:
            return self.__logpmf(x)
        elif self.__pmf is not None:
            return np.log(self.__pmf(x))
        else:
            raise NotImplementedError


class ContinuousRandomVariable(RandomVariable[_ValueType]):
    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType,
        random_state: Optional[RandomStateType] = None,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[ShapeArgType], _ValueType]] = None,
        in_support: Optional[Callable[[_ValueType], bool]] = None,
        pdf: Optional[Callable[[_ValueType], np.float_]] = None,
        logpdf: Optional[Callable[[_ValueType], np.float_]] = None,
        cdf: Optional[Callable[[_ValueType], np.float_]] = None,
        logcdf: Optional[Callable[[_ValueType], np.float_]] = None,
        quantile: Optional[Callable[[FloatArgType], _ValueType]] = None,
        mode: Optional[Callable[[], _ValueType]] = None,
        median: Optional[Callable[[], _ValueType]] = None,
        mean: Optional[Callable[[], _ValueType]] = None,
        cov: Optional[Callable[[], _ValueType]] = None,
        var: Optional[Callable[[], _ValueType]] = None,
        std: Optional[Callable[[], _ValueType]] = None,
        entropy: Optional[Callable[[], np.float_]] = None,
    ):
        # Probability density function
        self.__pdf = pdf
        self.__logpdf = logpdf

        super().__init__(
            shape=shape,
            dtype=dtype,
            random_state=random_state,
            parameters=parameters,
            sample=sample,
            in_support=in_support,
            cdf=cdf,
            logcdf=logcdf,
            quantile=quantile,
            mode=mode,
            median=median,
            mean=mean,
            cov=cov,
            var=var,
            std=std,
            entropy=entropy,
        )

    def pdf(self, x: _ValueType) -> np.float_:
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

    def logpdf(self, x: _ValueType) -> np.float_:
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
    >>> from probnum import asrandvar
    >>> bern = bernoulli(p=0.5)
    >>> bern.random_state = 42  # Seed for reproducibility
    >>> b = asrandvar(bern)
    >>> b.sample(size=5)
    array([1, 1, 1, 0, 0])
    """

    # pylint: disable=import-outside-toplevel,cyclic-import
    from probnum import random_variables as rvs

    # RandomVariable
    if isinstance(obj, RandomVariable):
        return obj
    # Scalar
    elif np.isscalar(obj):
        return rvs.Dirac(support=obj)
    # Numpy array, sparse array or Linear Operator
    elif isinstance(
        obj, (np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator)
    ):
        return rvs.Dirac(support=obj)
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
) -> RandomVariable:
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

    # pylint: disable=import-outside-toplevel,cyclic-import
    from probnum import random_variables as rvs

    # Univariate distributions (implemented in this package)
    if isinstance(scipyrv, scipy.stats._distn_infrastructure.rv_frozen):
        # Normal distribution
        if scipyrv.dist.name == "norm":
            return rvs.Normal(
                mean=scipyrv.mean(),
                cov=scipyrv.var(),
                random_state=scipyrv.random_state,
            )
    # Multivariate distributions (implemented in this package)
    elif isinstance(scipyrv, scipy.stats._multivariate.multi_rv_frozen):
        # Multivariate normal
        if scipyrv.__class__.__name__ == "multivariate_normal_frozen":
            return rvs.Normal(
                mean=scipyrv.mean,
                cov=scipyrv.cov,
                random_state=scipyrv.random_state,
            )

    # Generic distributions
    def _wrap_np_scalar(fn):
        if fn is None:
            return None

        def _wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)

            if np.isscalar(res):
                return _utils.as_numpy_scalar(res)

            return res

        return _wrapper

    if (
        hasattr(scipyrv, "dist") and isinstance(scipyrv.dist, scipy.stats.rv_discrete)
    ) or hasattr(scipyrv, "pmf"):
        rv_subclass = DiscreteRandomVariable
        rv_subclass_kwargs = {
            "pmf": _wrap_np_scalar(getattr(scipyrv, "pmf", None)),
            "logpmf": _wrap_np_scalar(getattr(scipyrv, "logpmf", None)),
        }
    else:
        rv_subclass = ContinuousRandomVariable
        rv_subclass_kwargs = {
            "pdf": _wrap_np_scalar(getattr(scipyrv, "pdf", None)),
            "logpdf": _wrap_np_scalar(getattr(scipyrv, "logpdf", None)),
        }

    if isinstance(scipyrv, scipy.stats._distn_infrastructure.rv_frozen):

        def in_support(x):
            low, high = scipyrv.support()

            return bool(low <= x <= high)

    else:
        in_support = None

    # Infer shape and dtype
    sample = _wrap_np_scalar(scipyrv.rvs)()

    return rv_subclass(
        shape=sample.shape,
        dtype=sample.dtype,
        random_state=getattr(scipyrv, "random_state", None),
        sample=_wrap_np_scalar(getattr(scipyrv, "rvs", None)),
        in_support=in_support,
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
        **rv_subclass_kwargs,
    )
