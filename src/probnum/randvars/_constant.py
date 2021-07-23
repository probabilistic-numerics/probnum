"""(Almost surely) constant random variables."""

from typing import Callable, TypeVar

import numpy as np

from probnum import config, linops
from probnum import utils as _utils
from probnum.typing import ArrayLikeGetitemArgType, ShapeArgType, ShapeType

from . import _random_variable

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


_ValueType = TypeVar("ValueType")


class Constant(_random_variable.DiscreteRandomVariable[_ValueType]):
    """Random variable representing a constant value.

    Discrete random variable which (with probability one) takes a constant value. The
    law / image measure of this random variable is given by the Dirac delta measure
    which equals one in its (atomic) support and zero everywhere else.

    This class has the useful property that arithmetic operations between a
    :class:`Constant` random variable and an arbitrary :class:`RandomVariable` represent
    the same arithmetic operation with a constant.

    Parameters
    ----------
    support
        Constant value taken by the random variable. Also the (atomic) support of the
        associated Dirac measure.

    See Also
    --------
    RandomVariable : Class representing random variables.

    Notes
    -----
    The Dirac measure formalizes the concept of a Dirac delta function as encountered in
    physics, where it is used to model a point mass. Another way to formalize this idea
    is to define the Dirac delta as a linear operator as is done in functional analysis.
    While related, this is not the view taken here.

    Examples
    --------
    >>> from probnum import randvars
    >>> import numpy as np
    >>> rv1 = randvars.Constant(support=0.)
    >>> rv2 = randvars.Constant(support=1.)
    >>> rv = rv1 + rv2
    >>> rng = np.random.default_rng(seed=42)
    >>> rv.sample(rng, size=5)
    array([1., 1., 1., 1., 1.])
    """

    def __init__(
        self,
        support: _ValueType,
    ):
        if np.isscalar(support):
            support = _utils.as_numpy_scalar(support)

        self._support = support

        support_floating = self._support.astype(
            np.promote_types(self._support.dtype, np.float_)
        )

        if config.lazy_linalg:
            zero_cov = (
                linops.Scaling(0.0, shape=((self._support.size, self._support.size)))
                if self._support.ndim > 0
                else np.zeros(shape=())
            )
        else:
            zero_cov = np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                support_floating,
                shape=(
                    (self._support.size, self._support.size)
                    if self._support.ndim > 0
                    else ()
                ),
            )

        super().__init__(
            shape=self._support.shape,
            dtype=self._support.dtype,
            parameters={"support": self._support},
            sample=self._sample,
            in_support=lambda x: np.all(x == self._support),
            pmf=lambda x: np.float_(1.0 if np.all(x == self._support) else 0.0),
            cdf=lambda x: np.float_(1.0 if np.all(x >= self._support) else 0.0),
            mode=lambda: self._support,
            median=lambda: support_floating,
            mean=lambda: support_floating,
            cov=lambda: zero_cov,
            var=lambda: np.zeros_like(support_floating),
        )

    @cached_property
    def cov_cholesky(self):
        # Pure utility attribute (it is zero anyway).
        # Make Constant behave more like Normal with zero covariance.
        return self.cov

    @property
    def support(self) -> _ValueType:
        """Constant value taken by the random variable."""
        return self._support

    def __getitem__(self, key: ArrayLikeGetitemArgType) -> "Constant":
        """(Advanced) indexing, masking and slicing.

        This method supports all modes of array indexing presented in

        https://numpy.org/doc/1.19/reference/arrays.indexing.html.

        Parameters
        ----------
        key : int or slice or ndarray or tuple of None, int, slice, or ndarray
            Indices, slice objects and/or boolean masks specifying which entries to keep
            while marginalizing over all other entries.
        """
        return Constant(support=self._support[key])

    def reshape(self, newshape: ShapeType) -> "Constant":
        return Constant(
            support=self._support.reshape(newshape),
        )

    def transpose(self, *axes: int) -> "Constant":
        return Constant(
            support=self._support.transpose(*axes),
        )

    def _sample(self, rng: np.random.Generator, size: ShapeArgType = ()) -> _ValueType:
        size = _utils.as_shape(size)

        if size == ():
            return self._support.copy()
        else:
            return np.tile(self._support, reps=size + (1,) * self.ndim)

    # Unary arithmetic operations

    def __neg__(self) -> "Constant":
        return Constant(
            support=-self.support,
        )

    def __pos__(self) -> "Constant":
        return Constant(
            support=+self.support,
        )

    def __abs__(self) -> "Constant":
        return Constant(
            support=abs(self.support),
        )

    # Binary arithmetic operations

    @staticmethod
    def _binary_operator_factory(
        operator: Callable[[_ValueType, _ValueType], _ValueType]
    ) -> Callable[["Constant", "Constant"], "Constant"]:
        def _constant_rv_binary_operator(
            constant_rv1: Constant, constant_rv2: Constant
        ) -> Constant:
            return Constant(
                support=operator(constant_rv1.support, constant_rv2.support),
            )

        return _constant_rv_binary_operator
