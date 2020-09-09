"""
Random Variables.

This module implements random variables. Random variables are the main in- and outputs
of probabilistic numerical methods.
"""

from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

import numpy as np

from probnum import utils as _utils
from probnum.type import (
    RandomStateType,
    ShapeType,
    # Argument Types
    DTypeArgType,
    FloatArgType,
    RandomStateArgType,
    ArrayLikeGetitemArgType,
    ShapeArgType,
)

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

    The internals of :class:`RandomVariable` objects are assumed to be constant over
    their whole lifecycle. This is due to the caches used to make certain computations
    more efficient. As a consequence, altering the internal state of a
    :class:`RandomVariable` (e.g. its mean, cov, sampling function, etc.) will result in
    undefined behavior. In particular, this should be kept in mind when subclassing
    :class:`RandomVariable` or any of its descendants.

    Parameters
    ----------
    shape :
        Shape of realizations of this random variable.
    dtype :
        Data type of realizations of this random variable. If ``object`` will be
        converted to ``numpy.dtype``.
    as_value_type :
        Function which can be used to transform user-supplied arguments, interpreted as
        realizations of this random variable, to an easy-to-process, normalized format.
        Will be called internally to transform the argument of functions like
        ``in_support``, ``cdf`` and ``logcdf``, ``pmf`` and ``logpmf`` (in
        :class:`DiscreteRandomVariable`), ``pdf`` and ``logpdf`` (in
        :class:`ContinuousRandomVariable`), and potentially by similar functions in
        subclasses.

        For instance, this method is useful if (``log``)``cdf`` and (``log``)``pdf``
        both only work on :class:`np.float_` arguments, but we still want the user to be
        able to pass Python :class:`float`. Then ``as_value_type`` should be set to
        something like ``lambda x: np.float64(x)``.

    See Also
    --------
    asrandvar : Transform into a :class:`RandomVariable`.

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
        as_value_type: Optional[Callable[[Any], _ValueType]] = None,
    ):
        # pylint: disable=too-many-arguments,too-many-locals
        """Create a new random variable."""
        self.__shape = _utils.as_shape(shape)

        # Data Types
        self.__dtype = np.dtype(dtype)
        self.__median_dtype = RandomVariable.infer_median_dtype(self.__dtype)
        self.__moment_dtype = RandomVariable.infer_moment_dtype(self.__dtype)

        self._random_state = _utils.as_random_state(random_state)

        # Probability distribution of the random variable
        self.__parameters = parameters.copy() if parameters is not None else {}

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

        # Utilities
        self.__as_value_type = as_value_type

    def __repr__(self) -> str:
        return f"<{self.shape} {self.__class__.__name__} with dtype={self.dtype}>"

    @property
    def shape(self) -> ShapeType:
        """Shape of realizations of the random variable."""
        return self.__shape

    @cached_property
    def ndim(self) -> int:
        return len(self.__shape)

    @cached_property
    def size(self) -> int:
        return int(np.prod(self.__shape))

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) a realization of this random variable."""
        return self.__dtype

    @property
    def median_dtype(self) -> np.dtype:
        """The dtype of the :attr:`median`. It will be set to the dtype arising from
        the multiplication of values with dtypes :attr:`dtype` and :class:`np.float_`.
        This is motivated by the fact that, even for discrete random variables, e.g.
        integer-valued random variables, the :attr:`median` might lie in between two
        values in which case these values are averaged. For example, a uniform random
        variable on :math:`\\{ 1, 2, 3, 4 \\}` will have a median of :math:`2.5`.
        """
        return self.__median_dtype

    @property
    def moment_dtype(self) -> np.dtype:
        """The dtype of any (function of a) moment of the random variable, e.g. its
        :attr:`mean`, :attr:`cov`, :attr:`var`, or :attr:`std`. It will be set to the
        dtype arising from the multiplication of values with dtypes :attr:`dtype`
        and :class:`np.float_`. This is motivated by the mathematical definition of a
        moment as a sum or an integral over products of probabilities and values of the
        random variable, which are represented as using the dtypes :class:`np.float_`
        and :attr:`dtype`, respectively.
        """
        return self.__moment_dtype

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
        return self.__parameters.copy()

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
            shape=self.__shape,
            dtype=self.__dtype,
        )

        # Make immutable
        if isinstance(mode, np.ndarray):
            mode.setflags(write=False)

        return mode

    @cached_property
    def median(self) -> _ValueType:
        """
        Median of the random variable.

        To learn about the dtype of the median, see :attr:`median_dtype`.

        Returns
        -------
        median : float
            The median of the distribution.
        """

        if self.__shape != ():
            raise NotImplementedError(
                "The median is only defined for scalar random variables."
            )

        median = self.__median()

        RandomVariable._check_property_value(
            "median",
            median,
            shape=self.__shape,
            dtype=self.__median_dtype,
        )

        # Make immutable
        if isinstance(median, np.ndarray):
            median.setflags(write=False)

        return median

    @cached_property
    def mean(self) -> _ValueType:
        """
        Mean :math:`\\mathbb{E}(X)` of the distribution.

        To learn about the dtype of the mean, see :attr:`moment_dtype`.

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
            shape=self.__shape,
            dtype=self.__moment_dtype,
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

        To learn about the dtype of the covariance, see :attr:`moment_dtype`.

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
            dtype=self.__moment_dtype,
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

        To learn about the dtype of the variance, see :attr:`moment_dtype`.

        Returns
        -------
        var : array-like
            The variance of the distribution.
        """
        if self.__var is None:
            try:
                var = np.diag(self.cov).reshape(self.__shape).copy()
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            var = self.__var()

        RandomVariable._check_property_value(
            "variance",
            var,
            shape=self.__shape,
            dtype=self.__moment_dtype,
        )

        # Make immutable
        if isinstance(var, np.ndarray):
            var.setflags(write=False)

        return var

    @cached_property
    def std(self) -> _ValueType:
        """
        Standard deviation of the distribution.

        To learn about the dtype of the standard deviation, see :attr:`moment_dtype`.

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
            shape=self.__shape,
            dtype=self.__moment_dtype,
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

        entropy = RandomVariable._ensure_numpy_float(
            "entropy", entropy, force_scalar=True
        )

        return entropy

    def in_support(self, x: _ValueType) -> bool:
        if self.__in_support is None:
            raise NotImplementedError

        in_support = self.__in_support(self._as_value_type(x))

        if not isinstance(in_support, bool):
            raise ValueError(
                f"The function `in_support` must return a `bool`, but its return value "
                f"is of type `{type(x)}`."
            )

        return in_support

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
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The cdf evaluation will be broadcast over all additional dimensions.

        Returns
        -------
        q : array-like
            Value of the cumulative density function at the given points.
        """
        if self.__cdf is not None:
            return RandomVariable._ensure_numpy_float(
                "cdf", self.__cdf(self._as_value_type(x))
            )
        elif self.__logcdf is not None:
            cdf = np.exp(self.logcdf(self._as_value_type(x)))

            assert isinstance(cdf, np.float_)

            return cdf
        else:
            raise NotImplementedError(
                f"Neither the `cdf` nor the `logcdf` of the random variable object "
                f"with type `{type(self).__name__}` is implemented."
            )

    def logcdf(self, x: _ValueType) -> np.float_:
        """
        Log-cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the cumulative distribution function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The logcdf evaluation will be broadcast over all additional dimensions.

        Returns
        -------
        q : array-like
            Value of the log-cumulative density function at the given points.
        """
        if self.__logcdf is not None:
            return RandomVariable._ensure_numpy_float(
                "logcdf", self.__logcdf(self._as_value_type(x))
            )
        elif self.__cdf is not None:
            logcdf = np.log(self.__cdf(x))

            assert isinstance(logcdf, np.float_)

            return logcdf
        else:
            raise NotImplementedError(
                f"Neither the `logcdf` nor the `cdf` of the random variable object "
                f"with type `{type(self).__name__}` is implemented."
            )

    def quantile(self, p: FloatArgType) -> _ValueType:
        """Quantile function.

        The quantile function :math:`Q \\colon [0, 1] \\to \\mathbb{R}` of a random
        variable :math:`X` is defined as
        :math:`Q(p) = \\inf\\{ x \\in \\mathbb{R} \\colon p \\le F_X(x) \\}`, where
        :math:`F_X \\colon \\mathbb{R} \\to [0, 1]` is the :meth:`cdf` of the random
        variable. From the definition it follows that the quantile function always
        returns values of the same dtype as the random variable. For instance, for a
        discrete distribution over the integers, the returned quantiles will also be
        integers. This means that, in general, :math:`Q(0.5)` is not equal to the
        :attr:`median` as it is defined in this class. See
        https://en.wikipedia.org/wiki/Quantile_function for more details and examples.
        """
        if self.__shape != ():
            raise NotImplementedError(
                "The quantile function is only defined for scalar random variables."
            )

        if self.__quantile is None:
            raise NotImplementedError

        try:
            p = _utils.as_numpy_scalar(p, dtype=np.floating)
        except TypeError as exc:
            raise TypeError(
                "The given argument `p` can not be cast to a `np.floating` object."
            ) from exc

        quantile = self.__quantile(p)

        if quantile.shape != self.__shape:
            raise ValueError(
                f"The quantile function should return values of the same shape as the "
                f"random variable, i.e. {self.__shape}, but it returned a value with "
                f"{quantile.shape}."
            )

        if quantile.dtype != self.__dtype:
            raise ValueError(
                f"The quantile function should return values of the same dtype as the "
                f"random variable, i.e. `{self.__dtype.name}`, but it returned a value "
                f"with dtype `{quantile.dtype.name}`."
            )

        return quantile

    def __getitem__(self, key: ArrayLikeGetitemArgType) -> "RandomVariable":
        return RandomVariable(
            shape=np.empty(shape=self.shape)[key].shape,
            dtype=self.dtype,
            random_state=_utils.derive_random_seed(self.random_state),
            sample=lambda size: self.sample(size)[key],
            mode=lambda: self.mode[key],
            mean=lambda: self.mean[key],
            var=lambda: self.var[key],
            std=lambda: self.std[key],
            entropy=lambda: self.entropy,
            as_value_type=self.__as_value_type,
        )

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
        newshape = _utils.as_shape(newshape)

        return RandomVariable(
            shape=newshape,
            dtype=self.dtype,
            random_state=_utils.derive_random_seed(self.random_state),
            sample=lambda size: self.sample(size).reshape(size + newshape),
            mode=lambda: self.mode.reshape(newshape),
            median=lambda: self.median.reshape(newshape),
            mean=lambda: self.mean.reshape(newshape),
            cov=lambda: self.cov,
            var=lambda: self.var.reshape(newshape),
            std=lambda: self.std.reshape(newshape),
            entropy=lambda: self.entropy,
            as_value_type=self.__as_value_type,
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
        return RandomVariable(
            shape=np.empty(shape=self.shape).transpose(*axes).shape,
            dtype=self.dtype,
            random_state=_utils.derive_random_seed(self.random_state),
            sample=lambda size: self.sample(size).transpose(*axes),
            mode=lambda: self.mode.transpose(*axes),
            median=lambda: self.median.transpose(*axes),
            mean=lambda: self.mean.transpose(*axes),
            cov=lambda: self.cov,
            var=lambda: self.var.transpose(*axes),
            std=lambda: self.std.transpose(*axes),
            entropy=lambda: self.entropy,
            as_value_type=self.__as_value_type,
        )

    T = property(transpose)

    # Unary arithmetic operations

    def __neg__(self) -> "RandomVariable":
        return RandomVariable(
            shape=self.shape,
            dtype=self.dtype,
            random_state=_utils.derive_random_seed(self.random_state),
            sample=lambda size: -self.sample(size=size),
            in_support=lambda x: self.in_support(-x),
            mode=lambda: -self.mode,
            median=lambda: -self.median,
            mean=lambda: -self.mean,
            cov=lambda: self.cov,
            var=lambda: self.var,
            std=lambda: self.std,
            as_value_type=self.__as_value_type,
        )

    def __pos__(self) -> "RandomVariable":
        return RandomVariable(
            shape=self.shape,
            dtype=self.dtype,
            random_state=_utils.derive_random_seed(self.random_state),
            sample=lambda size: +self.sample(size=size),
            in_support=lambda x: self.in_support(+x),
            mode=lambda: +self.mode,
            median=lambda: +self.median,
            mean=lambda: +self.mean,
            cov=lambda: self.cov,
            var=lambda: self.var,
            std=lambda: self.std,
            as_value_type=self.__as_value_type,
        )

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

    def __add__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import add

        return add(self, other)

    def __radd__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import add

        return add(other, self)

    def __sub__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import sub

        return sub(self, other)

    def __rsub__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import sub

        return sub(other, self)

    def __mul__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mul

        return mul(self, other)

    def __rmul__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mul

        return mul(other, self)

    def __matmul__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import matmul

        return matmul(other, self)

    def __truediv__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import truediv

        return truediv(self, other)

    def __rtruediv__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import truediv

        return truediv(other, self)

    def __floordiv__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import floordiv

        return floordiv(self, other)

    def __rfloordiv__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import floordiv

        return floordiv(other, self)

    def __mod__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mod

        return mod(self, other)

    def __rmod__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mod

        return mod(other, self)

    def __divmod__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import divmod_

        return divmod_(self, other)

    def __rdivmod__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import divmod_

        return divmod_(other, self)

    def __pow__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import pow_

        return pow_(self, other)

    def __rpow__(self, other: Any) -> "RandomVariable":
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import pow_

        return pow_(other, self)

    @staticmethod
    def infer_median_dtype(value_dtype: DTypeArgType) -> np.dtype:
        return RandomVariable.infer_moment_dtype(value_dtype)

    @staticmethod
    def infer_moment_dtype(value_dtype: DTypeArgType) -> np.dtype:
        return np.promote_types(value_dtype, np.float_)

    def _as_value_type(self, x: Any) -> _ValueType:
        if self.__as_value_type is not None:
            return self.__as_value_type(x)

        return x

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

    @classmethod
    def _ensure_numpy_float(
        cls, name: str, value: Any, force_scalar: bool = False
    ) -> Union[np.float_, np.ndarray]:
        if np.isscalar(value):
            if not isinstance(value, np.float_):
                try:
                    value = _utils.as_numpy_scalar(value, dtype=np.float_)
                except TypeError as err:
                    raise TypeError(
                        f"The function `{name}` specified via the constructor of "
                        f"`{cls.__name__}` must return a scalar value that can be "
                        f"converted to a `np.float_`, which is not possible for "
                        f"{value} of type {type(value)}."
                    ) from err
        elif not force_scalar:
            try:
                value = np.asarray(value, dtype=np.float_)
            except TypeError as err:
                raise TypeError(
                    f"The function `{name}` specified via the constructor of "
                    f"`{cls.__name__}` must return a value that can be converted "
                    f"to a `np.ndarray` of type `np.float_`, which is not possible "
                    f"for {value} of type {type(value)}."
                ) from err
        else:
            raise TypeError(
                f"The function `{name}` specified via the constructor of "
                f"`{cls.__name__}` must return a scalar value, but {value} of type "
                f"{type(value)} is not scalar."
            )

        assert isinstance(value, (np.float_, np.ndarray))

        return value


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
            return DiscreteRandomVariable._ensure_numpy_float("pmf", self.__pmf(x))
        elif self.__logpmf is not None:
            pmf = np.exp(self.__logpmf(x))

            assert isinstance(pmf, np.float_)

            return pmf
        else:
            raise NotImplementedError(
                f"Neither the `pmf` nor the `logpmf` of the discrete random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )

    def logpmf(self, x: _ValueType) -> np.float_:
        if self.__logpmf is not None:
            return DiscreteRandomVariable._ensure_numpy_float(
                "logpmf", self.__logpmf(self._as_value_type(x))
            )
        elif self.__pmf is not None:
            logpmf = np.log(self.__pmf(self._as_value_type(x)))

            assert isinstance(logpmf, np.float_)

            return logpmf
        else:
            raise NotImplementedError(
                f"Neither the `logpmf` nor the `pmf` of the discrete random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )


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

        Following the predominant convention in mathematics, we express pdfs with
        respect to the Lebesgue measure unless stated otherwise.

        Parameters
        ----------
        x : array-like
            Evaluation points of the probability density / mass function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The pdf evaluation will be broadcast over all additional dimensions.

        Returns
        -------
        p : array-like
            Value of the probability density / mass function at the given points.

        """
        if self.__pdf is not None:
            return ContinuousRandomVariable._ensure_numpy_float(
                "pdf", self.__pdf(self._as_value_type(x))
            )
        if self.__logpdf is not None:
            pdf = np.exp(self.__logpdf(self._as_value_type(x)))

            assert isinstance(pdf, np.float_)

            return pdf
        raise NotImplementedError(
            f"Neither the `pdf` nor the `logpdf` of the continuous random variable "
            f"object with type `{type(self).__name__}` is implemented."
        )

    def logpdf(self, x: _ValueType) -> np.float_:
        """
        Natural logarithm of the probability density function.

        Parameters
        ----------
        x : array-like
            Evaluation points of the log-probability density/mass function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The logpdf evaluation will be broadcast over all additional dimensions.

        Returns
        -------
        logp : array-like
            Value of the log-probability density / mass function at the given points.
        """
        if self.__logpdf is not None:
            return ContinuousRandomVariable._ensure_numpy_float(
                "logpdf", self.__logpdf(self._as_value_type(x))
            )
        elif self.__pdf is not None:
            logpdf = np.log(self.__pdf(self._as_value_type(x)))

            assert isinstance(logpdf, np.float_)

            return logpdf
        else:
            raise NotImplementedError(
                f"Neither the `logpdf` nor the `pdf` of the continuous random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )
