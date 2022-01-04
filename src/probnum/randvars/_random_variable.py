"""Random Variables."""

import functools
import operator
from functools import cached_property
from typing import Any, Callable, Dict, Optional

import numpy as np

from probnum import backend, utils as _utils
from probnum.typing import (
    ArrayIndicesLike,
    DTypeLike,
    ScalarType,
    SeedType,
    ShapeLike,
    ShapeType,
)

# pylint: disable="too-many-lines"


class RandomVariable:
    """Random variables represent uncertainty about a value.

    Random variables generalize multi-dimensional arrays by encoding uncertainty
    about the (numerical) quantity in question. Despite their name, they do not
    necessarily represent stochastic objects. Random variables are also the
    primary  in- and outputs of probabilistic numerical methods.

    Instances of :class:`RandomVariable` can be added, multiplied, etc. with
    arrays and linear operators. This may change their distribution and therefore
    not necessarily all previously available methods are retained.

    Parameters
    ----------
    shape :
        Shape of realizations of this random variable.
    dtype :
        Data type of realizations of this random variable. If of type :class:`object`
        the argument will be converted to :class:`numpy.dtype`.
    parameters :
        Parameters of the distribution of the random variable.
    sample :
        Callable implementing sampling from the random variable.
    in_support :
        Callable checking whether the random variable takes value ``x`` with non-zero
        probability, i.e. if ``x`` is in the support of its distribution.
    cdf :
        Cumulative distribution function of the random variable.
    logcdf :
        Log-transformed cumulative distribution function of the random variable.
    quantile :
        Quantile function of the random variable.
    mode :
        Mode of the random variable. Value of the random variable at which its
        probability density function or probability mass function takes its maximal
        value.
    mean :
        Expected value of the random variable.
    cov :
        Covariance of the random variable.
    var :
        (Element-wise) variance of the random variable.
    std :
        (Element-wise) standard deviation of the random variable.
    entropy :
        Information-theoretic entropy :math:`H(X)` of the random variable.

    See Also
    --------
    asrandvar : Transform into a :class:`RandomVariable`.
    DiscreteRandomVariable : A random variable with countable range.
    ContinuousRandomVariable : A random variable with uncountably infinite range.

    Notes
    -----
    The internals of :class:`RandomVariable` objects are assumed to be constant over
    their whole lifecycle. This is due to the caches used to make certain computations
    more efficient. As a consequence, altering the internal state of a
    :class:`RandomVariable` (e.g. its mean, cov, sampling function, etc.) will result in
    undefined behavior. In particular, this should be kept in mind when subclassing
    :class:`RandomVariable` or any of its descendants.

    Sampling from random variables with fixed seed is not stable with respect to the
    order of operations (such as slicing, masking, etc.). This means sampling from a
    random variable and then slicing the resulting array does not necessarily
    return the same result as slicing the random variable and sampling from the
    result. However, the random seed ensures that each sequence of operations will
    always result in the same output.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[SeedType, ShapeType], backend.ndarray]] = None,
        in_support: Optional[Callable[[backend.ndarray], bool]] = None,
        cdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        logcdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        quantile: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        mode: Optional[Callable[[], backend.ndarray]] = None,
        median: Optional[Callable[[], backend.ndarray]] = None,
        mean: Optional[Callable[[], backend.ndarray]] = None,
        cov: Optional[Callable[[], backend.ndarray]] = None,
        var: Optional[Callable[[], backend.ndarray]] = None,
        std: Optional[Callable[[], backend.ndarray]] = None,
        entropy: Optional[Callable[[], ScalarType]] = None,
    ):
        # pylint: disable=too-many-arguments,too-many-locals
        """Create a new random variable."""
        self.__shape = _utils.as_shape(shape)

        # Data Types
        self.__dtype = backend.asdtype(dtype)

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

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with shape={self.shape}, dtype"
            f"={self.dtype}>"
        )

    @property
    def shape(self) -> ShapeType:
        """Shape of realizations of the random variable."""
        return self.__shape

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions of realizations of the random variable."""
        return len(self.__shape)

    @cached_property
    def size(self) -> int:
        """Size of realizations of the random variable, defined as the product over all
        components of :attr:`shape`."""
        return functools.reduce(operator.mul, self.__shape, 1)

    @property
    def dtype(self) -> backend.dtype:
        """Data type of (elements of) a realization of this random variable."""
        return self.__dtype

    @cached_property
    def median_dtype(self) -> backend.dtype:
        r"""The dtype of the :attr:`median`.

        It will be set to the dtype arising from the multiplication of values with
        dtypes :attr:`dtype` and :class:`~probnum.backend.double`. This is motivated by
        the fact that, even for discrete random variables, e.g. integer-valued random
        variables, the :attr:`median` might lie in between two values in which case
        these values are averaged. For example, a uniform random variable on :math:`\{
        1, 2, 3, 4 \}` will have a median of :math:`2.5`.
        """
        return backend.promote_types(self.dtype, backend.double)

    @cached_property
    def expectation_dtype(self) -> backend.dtype:
        r"""The dtype of an expectation of (a function of) the random variable.

        For instance, the :attr:`mean`, :attr:`cov`, :attr:`var`, :attr:`std`, and
        :attr:`entropy` of the random variable will have this dtype. It will be set
        to the dtype arising from the multiplication of values with dtypes :attr:`dtype`
        and :class:`~probnum.backend.double`. This is motivated by the mathematical
        definition of an expectation as a sum or an integral over products of
        probabilities and values of the random variable, which are represented as using
        the dtypes :class:`~probnum.backend.double` and :attr:`dtype`, respectively.
        """
        return backend.promote_types(self.dtype, backend.double)

    @property
    def parameters(self) -> Dict[str, Any]:
        """Parameters of the associated probability distribution.

        The parameters of the probability distribution of the random
        variable, e.g. mean, variance, scale, rate, etc. stored in a
        ``dict``.
        """
        return self.__parameters.copy()

    @cached_property
    def mode(self) -> backend.ndarray:
        """Mode of the random variable."""
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
    def median(self) -> backend.ndarray:
        """Median of the random variable.

        To learn about the dtype of the median, see
        :attr:`median_dtype`.
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
            dtype=self.median_dtype,
        )

        # Make immutable
        if isinstance(median, np.ndarray):
            median.setflags(write=False)

        return median

    @cached_property
    def mean(self) -> backend.ndarray:
        r"""Mean :math:`\mathbb{E}(X)` of the random variable.

        To learn about the dtype of the mean, see :attr:`expectation_dtype`.
        """
        if self.__mean is None:
            raise NotImplementedError

        mean = self.__mean()

        RandomVariable._check_property_value(
            "mean",
            mean,
            shape=self.__shape,
            dtype=self.expectation_dtype,
        )

        # Make immutable
        if isinstance(mean, np.ndarray):
            mean.setflags(write=False)

        return mean

    @cached_property
    def cov(self) -> backend.ndarray:
        r"""Covariance :math:`\operatorname{Cov}(X) = \mathbb{E}( (X - \mathbb{E}(X))
        (X - \mathbb{E}(X))^\top )` of the random variable.

        To learn about the dtype of the covariance, see :attr:`expectation_dtype`.
        """
        if self.__cov is None:
            raise NotImplementedError

        cov = self.__cov()

        RandomVariable._check_property_value(
            "covariance",
            cov,
            shape=(self.size, self.size) if self.ndim > 0 else (),
            dtype=self.expectation_dtype,
        )

        # Make immutable
        if isinstance(cov, np.ndarray):
            cov.setflags(write=False)

        return cov

    @cached_property
    def var(self) -> backend.ndarray:
        r"""Variance :math:`\operatorname{Var}(X) = \mathbb{E}( (X - \mathbb{E}(X))^2 )`
        of the random variable.

        To learn about the dtype of the variance, see :attr:`expectation_dtype`.
        """
        if self.__var is None:
            try:
                var = backend.reshape(
                    backend.diag(self.cov),
                    self.__shape,
                ).copy()
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            var = self.__var()

        RandomVariable._check_property_value(
            "variance",
            var,
            shape=self.__shape,
            dtype=self.expectation_dtype,
        )

        # Make immutable
        if isinstance(var, np.ndarray):
            var.setflags(write=False)

        return var

    @cached_property
    def std(self) -> backend.ndarray:
        """Standard deviation of the random variable.

        To learn about the dtype of the standard deviation, see
        :attr:`expectation_dtype`.
        """
        if self.__std is None:
            std = backend.sqrt(self.var)
        else:
            std = self.__std()

        RandomVariable._check_property_value(
            "standard deviation",
            std,
            shape=self.__shape,
            dtype=self.expectation_dtype,
        )

        # Make immutable
        if isinstance(std, np.ndarray):
            std.setflags(write=False)

        return std

    @cached_property
    def entropy(self) -> ScalarType:
        r"""Information-theoretic entropy :math:`H(X)` of the random variable."""
        if self.__entropy is None:
            raise NotImplementedError

        entropy = self.__entropy()

        RandomVariable._check_property_value(
            "entropy",
            value=entropy,
            shape=(),
            dtype=self.expectation_dtype,
        )

        return entropy

    def in_support(self, x: backend.ndarray) -> backend.ndarray:
        """Check whether the random variable takes value ``x`` with non-zero
        probability, i.e. if ``x`` is in the support of its distribution.

        Parameters
        ----------
        x :
            Input value.
        """
        if self.__in_support is None:
            raise NotImplementedError

        in_support = self.__in_support(backend.asarray(x))

        self._check_return_value(
            "in_support",
            input_value=x,
            return_value=in_support,
            expected_shape=x.shape[: -self.ndim],
            expected_dtype=backend.bool,
        )

        return in_support

    def sample(self, seed: SeedType, sample_shape: ShapeLike = ()) -> backend.ndarray:
        """Draw realizations from a random variable.

        Parameters
        ----------
        seed
            Seed used for sampling from a random number generator.
        sample_shape
            Size of the drawn sample of realizations.
        """
        if self.__sample is None:
            raise NotImplementedError("No sampling method provided.")

        samples = self.__sample(seed, _utils.as_shape(sample_shape))

        # TODO: Check shape and dtype

        return samples

    def cdf(self, x: backend.ndarray) -> backend.ndarray:
        """Cumulative distribution function.

        Parameters
        ----------
        x :
            Evaluation points of the cumulative distribution function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The cdf evaluation will be broadcast over all additional dimensions.
        """
        if self.__cdf is not None:
            cdf = self.__cdf(backend.asarray(x))
        elif self.__logcdf is not None:
            cdf = backend.exp(self.logcdf(x))
        else:
            raise NotImplementedError(
                f"Neither the `cdf` nor the `logcdf` of the random variable object "
                f"with type `{type(self).__name__}` is implemented."
            )

        self._check_return_value(
            "cdf",
            input_value=x,
            return_value=cdf,
            expected_shape=x.shape[: -self.ndim],
            expected_dtype=backend.double,
        )

        return cdf

    def logcdf(self, x: backend.ndarray) -> backend.ndarray:
        """Log-cumulative distribution function.

        Parameters
        ----------
        x :
            Evaluation points of the cumulative distribution function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The logcdf evaluation will be broadcast over all additional dimensions.
        """
        if self.__logcdf is not None:
            logcdf = self.__logcdf(backend.asarray(x))
        elif self.__cdf is not None:
            logcdf = backend.log(self.cdf(x))
        else:
            raise NotImplementedError(
                f"Neither the `logcdf` nor the `cdf` of the random variable object "
                f"with type `{type(self).__name__}` is implemented."
            )

        self._check_return_value(
            "logcdf",
            input_value=x,
            return_value=logcdf,
            expected_shape=x.shape[: -self.ndim],
            expected_dtype=backend.double,
        )

        return logcdf

    def quantile(self, p: backend.ndarray) -> backend.ndarray:
        r"""Quantile function.

        The quantile function :math:`Q \colon [0, 1] \to \mathbb{R}` of a random
        variable :math:`X` is defined as :math:`Q(p) = \inf \{ x \in \mathbb{R} \colon p
        \le F_X(x) \}`, where :math:`F_X \colon \mathbb{R} \to [0, 1]` is the
        :meth:`cdf` of the random variable. From the definition it follows that the
        quantile function always returns values of the same dtype as the random
        variable. For instance, for a discrete distribution over the integers, the
        returned quantiles will also be integers. This means that, in general,
        :math:`Q(0.5)` is not equal to the :attr:`median` as it is defined in this
        class. See https://en.wikipedia.org/wiki/Quantile_function for more details and
        examples.
        """
        if self.__shape != ():
            raise NotImplementedError(
                "The quantile function is only defined for scalar random variables."
            )

        if self.__quantile is None:
            raise NotImplementedError

        quantile = self.__quantile(p)

        self._check_return_value(
            "quantile",
            input_value=p,
            return_value=quantile,
            expected_shape=p.shape + self.shape,
            expected_dtype=self.dtype,
        )

        return quantile

    def __getitem__(self, key: ArrayIndicesLike) -> "RandomVariable":
        # Shape inference
        # For simplicity, this should not be computed using backend, but rather in numpy
        shape = np.broadcast_to(np.empty(()), self.shape)[key].shape

        return RandomVariable(
            shape=shape,
            dtype=self.dtype,
            sample=lambda rng, size: self.sample(rng, size)[key],
            mode=lambda: self.mode[key],
            mean=lambda: self.mean[key],
            var=lambda: self.var[key],
            std=lambda: self.std[key],
            entropy=lambda: self.entropy,
        )

    def reshape(self, newshape: ShapeLike) -> "RandomVariable":
        """Give a new shape to a random variable.

        Parameters
        ----------
        newshape :
            New shape for the random variable. It must be compatible with the original
            shape.
        """
        newshape = _utils.as_shape(newshape)

        return RandomVariable(
            shape=newshape,
            dtype=self.dtype,
            sample=lambda rng, size: self.sample(rng, size).reshape(size + newshape),
            mode=lambda: self.mode.reshape(newshape),
            median=lambda: self.median.reshape(newshape),
            mean=lambda: self.mean.reshape(newshape),
            cov=lambda: self.cov,
            var=lambda: self.var.reshape(newshape),
            std=lambda: self.std.reshape(newshape),
            entropy=lambda: self.entropy,
        )

    def transpose(self, *axes: int) -> "RandomVariable":
        """Transpose the random variable.

        Parameters
        ----------
        axes :
            See documentation of :meth:`numpy.ndarray.transpose`.
        """

        # Shape inference
        # For simplicity, this should not be computed using backend, but rather in numpy
        shape = np.broadcast_to(np.empty(()), self.shape).transpose(*axes).shape

        return RandomVariable(
            shape=shape,
            dtype=self.dtype,
            sample=lambda rng, size: self.sample(rng, size).transpose(*axes),
            mode=lambda: self.mode.transpose(*axes),
            median=lambda: self.median.transpose(*axes),
            mean=lambda: self.mean.transpose(*axes),
            cov=lambda: self.cov,
            var=lambda: self.var.transpose(*axes),
            std=lambda: self.std.transpose(*axes),
            entropy=lambda: self.entropy,
        )

    T = property(transpose)

    # Unary arithmetic operations

    def __neg__(self) -> "RandomVariable":
        return RandomVariable(
            shape=self.shape,
            dtype=self.dtype,
            sample=lambda seed, sample_shape: -self.sample(
                seed=seed, sample_shape=sample_shape
            ),
            in_support=lambda x: self.in_support(-x),
            mode=lambda: -self.mode,
            median=lambda: -self.median,
            mean=lambda: -self.mean,
            cov=lambda: self.cov,
            var=lambda: self.var,
            std=lambda: self.std,
        )

    def __pos__(self) -> "RandomVariable":
        return RandomVariable(
            shape=self.shape,
            dtype=self.dtype,
            sample=lambda seed, sample_shape: +self.sample(
                seed=seed, sample_shape=sample_shape
            ),
            in_support=lambda x: self.in_support(+x),
            mode=lambda: +self.mode,
            median=lambda: +self.median,
            mean=lambda: +self.mean,
            cov=lambda: self.cov,
            var=lambda: self.var,
            std=lambda: self.std,
        )

    def __abs__(self) -> "RandomVariable":
        return RandomVariable(
            shape=self.shape,
            dtype=self.dtype,
            sample=lambda seed, sample_shape: abs(
                self.sample(seed=seed, sample_shape=sample_shape)
            ),
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
    def _check_property_value(
        name: str,
        value: backend.ndarray,
        shape: Optional[ShapeType] = None,
        dtype: Optional[backend.dtype] = None,
    ):
        if shape is not None:
            if value.shape != shape:
                raise ValueError(
                    f"The {name} of the random variable does not have the correct "
                    f"shape. Expected {shape} but got {value.shape}."
                )

        if dtype is not None:
            if value.dtype != dtype:
                raise ValueError(
                    f"The {name} of the random variable does not have the correct "
                    f"dtype. Expected {str(dtype)} but got {str(value.dtype)}."
                )

    def _check_return_value(
        self,
        method_name: str,
        input_value: backend.ndarray,
        return_value: backend.ndarray,
        expected_shape: Optional[ShapeType] = None,
        expected_dtype: Optional[backend.dtype] = None,
    ):
        # pylint: disable=too-many-arguments

        if expected_shape is not None:
            if return_value.shape != expected_shape:
                raise ValueError(
                    f"The return value of the function `{method_name}` does not have "
                    f"the correct shape for an input with shape {input_value.shape} "
                    f"and a random variable with shape {self.shape}. Expected "
                    f"{expected_shape} but got {return_value.shape}."
                )

        if expected_dtype is not None:
            if return_value.dtype != expected_dtype:
                raise ValueError(
                    f"The return value of the function `{method_name}` does not have "
                    f"the correct dtype for an input with dtype "
                    f"{str(input_value.dtype)} and a random variable with dtype "
                    f"{str(self.dtype)}. Expected {str(expected_dtype)} but got "
                    f"{str(return_value.dtype)}."
                )


class DiscreteRandomVariable(RandomVariable):
    """Random variable with countable range.

    Discrete random variables map to a countable set. Typical examples are the natural
    numbers or integers.

    Parameters
    ----------
    shape :
        Shape of realizations of this random variable.
    dtype :
        Data type of realizations of this random variable. If ``object`` will be
        converted to ``numpy.dtype``.
    parameters :
        Parameters of the distribution of the random variable.
    sample :
        Callable implementing sampling from the random variable.
    in_support :
        Callable checking whether the random variable takes value ``x`` with non-zero
        probability, i.e. if ``x`` is in the support of its distribution.
    pmf :
        Probability mass function of the random variable.
    logpmf :
        Log-transformed probability mass function of the random variable.
    cdf :
        Cumulative distribution function of the random variable.
    logcdf :
        Log-transformed cumulative distribution function of the random variable.
    quantile :
        Quantile function of the random variable.
    mode :
        Mode of the random variable. Value of the random variable at which
        :meth:`~DiscreteRandomVariable.pmf` takes its maximal value.
    mean :
        Expected value of the random variable.
    cov :
        Covariance of the random variable.
    var :
        (Element-wise) variance of the random variable.
    std :
        (Element-wise) standard deviation of the random variable.
    entropy :
        Shannon entropy :math:`H(X)` of the random variable.

    See Also
    --------
    RandomVariable : Class representing random variables.
    ContinuousRandomVariable : A random variable with uncountably infinite range.

    Examples
    --------
    >>> # Create a custom categorical random variable
    >>> import numpy as np
    >>> from probnum.randvars import DiscreteRandomVariable
    >>>
    >>> # Distribution parameters
    >>> support = np.array([-1, 0, 1])
    >>> p = np.array([0.2, 0.5, 0.3])
    >>> parameters_categorical = {
    ...     "support" : support,
    ...     "p" : p}
    >>>
    >>> # Sampling function
    >>> def sample_categorical(rng, size=()):
    ...     return rng.choice(a=support, size=size, p=p)
    >>>
    >>> # Probability mass function
    >>> def pmf_categorical(x):
    ...     idx = np.where(x == support)[0]
    ...     if len(idx) > 0:
    ...         return p[idx]
    ...     else:
    ...         return 0.0
    >>>
    >>> # Create custom random variable
    >>> x = DiscreteRandomVariable(
    ...       shape=(),
    ...       dtype=np.dtype(np.int64),
    ...       parameters=parameters_categorical,
    ...       sample=sample_categorical,
    ...       pmf=pmf_categorical,
    ...       mean=lambda : np.float64(0),
    ...       median=lambda : np.float64(0),
    ...       )
    >>>
    >>> # Sample from new random variable
    >>> rng = np.random.default_rng(42)
    >>> x.sample(rng=rng, size=3)
    array([1, 0, 1])
    >>> x.pmf(2)
    array(0.)
    >>> x.mean
    0.0
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[SeedType, ShapeType], backend.ndarray]] = None,
        in_support: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        pmf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        logpmf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        cdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        logcdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        quantile: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        mode: Optional[Callable[[], backend.ndarray]] = None,
        median: Optional[Callable[[], backend.ndarray]] = None,
        mean: Optional[Callable[[], backend.ndarray]] = None,
        cov: Optional[Callable[[], backend.ndarray]] = None,
        var: Optional[Callable[[], backend.ndarray]] = None,
        std: Optional[Callable[[], backend.ndarray]] = None,
        entropy: Optional[Callable[[], ScalarType]] = None,
    ):
        # pylint: disable=too-many-arguments,too-many-locals

        # Probability mass function
        self.__pmf = pmf
        self.__logpmf = logpmf

        super().__init__(
            shape=shape,
            dtype=dtype,
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

    def pmf(self, x: backend.ndarray) -> backend.ndarray:
        """Probability mass function.

        Computes the probability of the random variable being equal to the given
        value. For a random variable :math:`X` it is defined as
        :math:`p_X(x) = P(X = x)` for a probability measure :math:`P`.

        Probability mass functions are the discrete analogue of probability density
        functions in the sense that they are the Radon-Nikodym derivative of the
        pushforward measure :math:`P \\circ X^{-1}` defined by the random variable with
        respect to the counting measure.

        Parameters
        ----------
        x :
            Evaluation points of the probability mass function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The pmf evaluation will be broadcast over all additional dimensions.
        """
        if self.__pmf is not None:
            pmf = self.__pmf(backend.asarray(x))
        elif self.__logpmf is not None:
            pmf = backend.exp(self.logpmf(x))
        else:
            raise NotImplementedError(
                f"Neither the `pmf` nor the `logpmf` of the discrete random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )

        self._check_return_value(
            "pmf",
            input_value=x,
            return_value=pmf,
            expected_shape=x.shape[: -self.ndim],
            expected_dtype=backend.double,
        )

        return pmf

    def logpmf(self, x: backend.ndarray) -> backend.ndarray:
        """Natural logarithm of the probability mass function.

        Parameters
        ----------
        x :
            Evaluation points of the log-probability mass function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The logpmf evaluation will be broadcast over all additional dimensions.
        """
        if self.__logpmf is not None:
            logpmf = self.__logpmf(backend.asarray(x))
        elif self.__pmf is not None:
            logpmf = backend.log(self.pmf(x))
        else:
            raise NotImplementedError(
                f"Neither the `logpmf` nor the `pmf` of the discrete random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )

        self._check_return_value(
            "logpmf",
            input_value=x,
            return_value=logpmf,
            expected_shape=x.shape[: -self.ndim],
            expected_dtype=backend.double,
        )

        return logpmf


class ContinuousRandomVariable(RandomVariable):
    """Random variable with uncountably infinite range.

    Continuous random variables map to a uncountably infinite set. Typically, this is a
    subset of a real vector space.

    Parameters
    ----------
    shape :
        Shape of realizations of this random variable.
    dtype :
        Data type of realizations of this random variable. If ``object`` will be
        converted to ``numpy.dtype``.
    parameters :
        Parameters of the distribution of the random variable.
    sample :
        Callable implementing sampling from the random variable.
    in_support :
        Callable checking whether the random variable takes value ``x`` with non-zero
        probability, i.e. if ``x`` is in the support of its distribution.
    pdf :
        Probability density function of the random variable.
    logpdf :
        Log-transformed probability density function of the random variable.
    cdf :
        Cumulative distribution function of the random variable.
    logcdf :
        Log-transformed cumulative distribution function of the random variable.
    quantile :
        Quantile function of the random variable.
    mode :
        Mode of the random variable. Value of the random variable at which
        :meth:`~ContinuousRandomVariable.pdf` takes its maximal value.
    mean :
        Expected value of the random variable.
    cov :
        Covariance of the random variable.
    var :
        (Element-wise) variance of the random variable.
    std :
        (Element-wise) standard deviation of the random variable.
    entropy :
        Differential entropy :math:`H(X)` of the random variable.

    See Also
    --------
    RandomVariable : Class representing random variables.
    DiscreteRandomVariable : A random variable with countable range.

    Examples
    --------
    >>> # Create a custom uniformly distributed random variable
    >>> import numpy as np
    >>>
    >>> # Distribution parameters
    >>> a = 0.0
    >>> b = 1.0
    >>> parameters_uniform = {"bounds" : [a, b]}
    >>>
    >>> # Sampling function
    >>> def sample_uniform(rng, size=()):
    ...     return rng.uniform(size=size)
    >>>
    >>> # Probability density function
    >>> def pdf_uniform(x):
    ...     if a <= x < b:
    ...         return 1.0
    ...     else:
    ...         return 0.0
    >>>
    >>> # Create custom random variable
    >>> u = ContinuousRandomVariable(
    ...       shape=(),
    ...       dtype=np.dtype(np.float64),
    ...       parameters=parameters_uniform,
    ...       sample=sample_uniform,
    ...       pdf=pdf_uniform,
    ...       mean=lambda : np.float64(0.5 * (a + b)),
    ...       median=lambda : np.float64(0.5 * (a + b)),
    ...       var=lambda : np.float64(1 / 12 * (b - a) ** 2),
    ...       entropy=lambda : np.log(b - a),
    ...       )
    >>>
    >>> # Sample from new random variable
    >>> rng = np.random.default_rng(42)
    >>> u.sample(rng=rng, size=3)
    array([0.77395605, 0.43887844, 0.85859792])
    >>> u.pdf(0.5)
    array(1.)
    >>> u.var
    0.08333333333333333
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
        parameters: Optional[Dict[str, Any]] = None,
        sample: Optional[Callable[[SeedType, ShapeType], backend.ndarray]] = None,
        in_support: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        pdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        logpdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        cdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        logcdf: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        quantile: Optional[Callable[[backend.ndarray], backend.ndarray]] = None,
        mode: Optional[Callable[[], backend.ndarray]] = None,
        median: Optional[Callable[[], backend.ndarray]] = None,
        mean: Optional[Callable[[], backend.ndarray]] = None,
        cov: Optional[Callable[[], backend.ndarray]] = None,
        var: Optional[Callable[[], backend.ndarray]] = None,
        std: Optional[Callable[[], backend.ndarray]] = None,
        entropy: Optional[Callable[[], backend.ndarray]] = None,
    ):
        # pylint: disable=too-many-arguments,too-many-locals

        # Probability density function
        self.__pdf = pdf
        self.__logpdf = logpdf

        super().__init__(
            shape=shape,
            dtype=dtype,
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

    def pdf(self, x: backend.ndarray) -> backend.ndarray:
        """Probability density function.

        The area under the curve defined by the probability density function
        specifies the probability of the random variable :math:`X` taking values
        within that area.

        Probability density functions are defined as the Radon-Nikodym derivative of
        the pushforward measure :math:`P \\circ X^{-1}` with respect to the Lebesgue
        measure for a given probability measure :math:`P`. Following convention we
        always assume the Lebesgue measure as a base measure unless stated otherwise.

        Parameters
        ----------
        x :
            Evaluation points of the probability density function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The pdf evaluation will be broadcast over all additional dimensions.
        """
        if self.__pdf is not None:
            pdf = self.__pdf(backend.asarray(x))
        elif self.__logpdf is not None:
            pdf = backend.exp(self.logpdf(x))
        else:
            raise NotImplementedError(
                f"Neither the `pdf` nor the `logpdf` of the continuous random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )

        self._check_return_value(
            "pdf",
            input_value=x,
            return_value=pdf,
            expected_shape=x.shape[: x.ndim - self.ndim],
            expected_dtype=backend.double,
        )

        return pdf

    def logpdf(self, x: backend.ndarray) -> backend.ndarray:
        """Natural logarithm of the probability density function.

        Parameters
        ----------
        x :
            Evaluation points of the log-probability density function.
            The shape of this argument should be :code:`(..., S1, ..., SN)`, where
            :code:`(S1, ..., SN)` is the :attr:`shape` of the random variable.
            The logpdf evaluation will be broadcast over all additional dimensions.
        """
        if self.__logpdf is not None:
            logpdf = self.__logpdf(backend.asarray(x))
        elif self.__pdf is not None:
            logpdf = backend.log(self.pdf(x))
        else:
            raise NotImplementedError(
                f"Neither the `logpdf` nor the `pdf` of the continuous random variable "
                f"object with type `{type(self).__name__}` is implemented."
            )

        self._check_return_value(
            "logpdf",
            input_value=x,
            return_value=logpdf,
            expected_shape=x.shape[: -self.ndim],
            expected_dtype=backend.double,
        )

        return logpdf
