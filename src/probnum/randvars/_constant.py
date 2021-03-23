"""(Almost surely) constant random variables."""

from typing import Callable, TypeVar

import numpy as np

from probnum import utils as _utils
from probnum.type import (
    ArrayLikeGetitemArgType,
    RandomStateArgType,
    ShapeArgType,
    ShapeType,
)

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
    random_state
        Random state of the random variable. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

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
    >>> rv1 = randvars.Constant(support=0.)
    >>> rv2 = randvars.Constant(support=1.)
    >>> rv = rv1 + rv2
    >>> rv.sample(size=5)
    array([1., 1., 1., 1., 1.])
    """

    def __init__(
        self,
        support: _ValueType,
        random_state: RandomStateArgType = None,
    ):
        if np.isscalar(support):
            support = _utils.as_numpy_scalar(support)

        self._support = support

        support_floating = self._support.astype(
            np.promote_types(self._support.dtype, np.float_)
        )

        super().__init__(
            shape=self._support.shape,
            dtype=self._support.dtype,
            random_state=random_state,
            parameters={"support": self._support},
            sample=self._sample,
            in_support=lambda x: np.all(x == self._support),
            pmf=lambda x: np.float_(1.0 if np.all(x == self._support) else 0.0),
            cdf=lambda x: np.float_(1.0 if np.all(x >= self._support) else 0.0),
            mode=lambda: self._support,
            median=lambda: support_floating,
            mean=lambda: support_floating,
            cov=lambda: np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                support_floating,
                shape=(
                    (self._support.size, self._support.size)
                    if self._support.ndim > 0
                    else ()
                ),
            ),
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
        return Constant(support=self._support[key], random_state=self.random_state)

    def reshape(self, newshape: ShapeType) -> "Constant":
        return Constant(
            support=self._support.reshape(newshape),
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def transpose(self, *axes: int) -> "Constant":
        return Constant(
            support=self._support.transpose(*axes),
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def _sample(self, size: ShapeArgType = ()) -> _ValueType:
        size = _utils.as_shape(size)

        if size == ():
            return self._support.copy()
        else:
            return np.tile(self._support, reps=size + (1,) * self.ndim)

    # Unary arithmetic operations

    def __neg__(self) -> "Constant":
        return Constant(
            support=-self.support,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __pos__(self) -> "Constant":
        return Constant(
            support=+self.support,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __abs__(self) -> "Constant":
        return Constant(
            support=abs(self.support),
            random_state=_utils.derive_random_seed(self.random_state),
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
                random_state=_utils.derive_random_seed(
                    constant_rv1.random_state,
                    constant_rv2.random_state,
                ),
            )

        return _constant_rv_binary_operator
