"""Random variables that are constant (with probability one)."""

from __future__ import annotations

from functools import cached_property
from typing import Callable

from probnum import backend, config, linops
from probnum.backend.random import RNGState
from probnum.backend.typing import ArrayIndicesLike, ShapeLike, ShapeType

from . import _random_variable


class Constant(_random_variable.DiscreteRandomVariable):
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
        support: backend.Array,
    ):
        self._support = backend.asarray(support)

        support_floating = self._support.astype(
            backend.promote_types(self._support.dtype, backend.float64)
        )

        if config.matrix_free:
            cov = lambda: (
                linops.Zero(shape=((self._support.size, self._support.size)))
                if self._support.ndim > 0
                else backend.asscalar(0.0, support_floating.dtype)
            )
        else:
            cov = lambda: backend.broadcast_to(
                backend.asscalar(0.0, support_floating.dtype),
                shape=(
                    (self._support.size, self._support.size)
                    if self._support.ndim > 0
                    else ()
                ),
            )

        var = lambda: backend.broadcast_to(
            backend.asscalar(0.0, support_floating.dtype),
            shape=self._support.shape,
        )

        super().__init__(
            shape=self._support.shape,
            dtype=self._support.dtype,
            parameters={"support": self._support},
            sample=self._sample,
            in_support=lambda x: backend.all(x == self._support),
            pmf=lambda x: backend.float64(
                1.0 if backend.all(x == self._support) else 0.0
            ),
            cdf=lambda x: backend.float64(
                1.0 if backend.all(x >= self._support) else 0.0
            ),
            mode=lambda: self._support,
            median=lambda: support_floating,
            mean=lambda: support_floating,
            cov=cov,
            var=var,
            std=var,
        )

    @cached_property
    def _cov_cholesky(self):
        # Pure utility attribute (it is zero anyway).
        # Make Constant behave more like Normal with zero covariance.
        return self.cov

    @property
    def support(self) -> backend.Array:
        """Constant value taken by the random variable."""
        return self._support

    def __getitem__(self, key: ArrayIndicesLike) -> "Constant":
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

    def _sample(
        self, rng_state: RNGState, sample_shape: ShapeLike = ()
    ) -> backend.Array:
        # pylint: disable=unused-argument

        if sample_shape == ():
            return self._support.copy()

        return backend.tile(self._support, reps=sample_shape + (1,) * self.ndim)

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
        operator: Callable[[backend.Array, backend.Array], backend.Array]
    ) -> Callable[["Constant", "Constant"], "Constant"]:
        def _constant_rv_binary_operator(
            constant_rv1: Constant, constant_rv2: Constant
        ) -> Constant:
            return Constant(
                support=operator(constant_rv1.support, constant_rv2.support),
            )

        return _constant_rv_binary_operator
