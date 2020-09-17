""" This module implements Dirac-distributed random variables. """

from typing import Callable, TypeVar

import numpy as np

from probnum import utils as _utils
from probnum.type import (
    ShapeType,
    # Argument Types
    ArrayLikeGetitemArgType,
    RandomStateArgType,
    ShapeArgType,
)

from . import _random_variable


_ValueType = TypeVar("ValueType")


class Dirac(_random_variable.DiscreteRandomVariable[_ValueType]):
    """
    The Dirac delta distribution.

    This distribution models a point mass and can be useful to represent
    numbers as random variables with Dirac measure. It has the useful
    property that arithmetic operations between a :class:`Dirac` random
    variable and an arbitrary :class:`RandomVariable` acts in the same
    way as the arithmetic operation with a constant.

    Note, that a Dirac measure does not admit a probability density
    function but can be viewed as a distribution (generalized function).

    Parameters
    ----------
    support : scalar or array-like or LinearOperator
        The support of the dirac delta function.

    See Also
    --------
    RandomVariable : Class representing general random variables.

    Examples
    --------
    >>> from probnum import random_variables as rvs
    >>> rv1 = rvs.Dirac(support=0.)
    >>> rv2 = rvs.Dirac(support=1.)
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

    @property
    def support(self) -> _ValueType:
        return self._support

    def __getitem__(self, key: ArrayLikeGetitemArgType) -> "Dirac":
        """
        Marginalization for multivariate Dirac distributions, expressed by means of
        (advanced) indexing, masking and slicing.

        This method supports all modes of array indexing presented in

        https://numpy.org/doc/1.19/reference/arrays.indexing.html.

        Parameters
        ----------
        key : int or slice or ndarray or tuple of None, int, slice, or ndarray
            Indices, slice objects and/or boolean masks specifying which entries to keep
            while marginalizing over all other entries.
        """
        return Dirac(support=self._support[key], random_state=self.random_state)

    def reshape(self, newshape: ShapeType) -> "Dirac":
        return Dirac(
            support=self._support.reshape(newshape),
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def transpose(self, *axes: int) -> "Dirac":
        return Dirac(
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

    def __neg__(self) -> "Dirac":
        return Dirac(
            support=-self.support,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __pos__(self) -> "Dirac":
        return Dirac(
            support=+self.support,
            random_state=_utils.derive_random_seed(self.random_state),
        )

    def __abs__(self) -> "Dirac":
        return Dirac(
            support=abs(self.support),
            random_state=_utils.derive_random_seed(self.random_state),
        )

    # Binary arithmetic operations

    @staticmethod
    def _binary_operator_factory(
        operator: Callable[[_ValueType, _ValueType], _ValueType]
    ) -> Callable[["Dirac", "Dirac"], "Dirac"]:
        def _dirac_binary_operator(dirac_rv1: Dirac, dirac_rv2: Dirac) -> Dirac:
            return Dirac(
                support=operator(dirac_rv1.support, dirac_rv2.support),
                random_state=_utils.derive_random_seed(
                    dirac_rv1.random_state,
                    dirac_rv2.random_state,
                ),
            )

        return _dirac_binary_operator
