"""
Dirac delta distribution.
"""
import operator

import numpy as np

from probnum.prob.distributions.distribution import Distribution


class Dirac(Distribution):
    """
    The Dirac delta distribution.

    This distribution models a point mass and can be useful to represent numbers as random variables with Dirac measure.
    It has the useful property that arithmetic operations between a :class:`Dirac` random variable and an arbitrary
    :class:`RandomVariable` acts in the same way as the arithmetic operation with a constant.

    Note, that a Dirac measure does not admit a probability density function but can be viewed as a distribution
    (generalized function).

    Parameters
    ----------
    support : scalar or array-like or LinearOperator
        The support of the dirac delta function.

    See Also
    --------
    Distribution : Class representing general probability distribution.

    Examples
    --------
    >>> from probnum.prob import RandomVariable, Dirac
    >>> dist1 = Dirac(support=0.)
    >>> dist2 = Dirac(support=1.)
    >>> rv = RandomVariable(distribution=dist1 + dist2)
    >>> rv.sample(size=5)
    array([1., 1., 1., 1., 1.])
    """

    def __init__(self, support, random_state=None):
        # Set dtype
        if np.isscalar(support):
            _dtype = np.dtype(type(support))
        else:
            _dtype = support.dtype

        # Initializer of superclass
        super().__init__(parameters={"support": support}, dtype=_dtype, random_state=random_state)

    def cdf(self, x):
        if np.any(x < self.parameters["support"]):
            return 0.
        else:
            return 1.

    def median(self):
        return self.parameters["support"]

    def mode(self):
        return self.parameters["support"]

    def mean(self):
        return self.parameters["support"]

    def var(self):
        return 0.

    def sample(self, size=(), seed=None):
        ndims = len(self.shape)
        if size == 1 or size == ():
            return self.parameters["support"]
        elif isinstance(size, int) and ndims == 0:
            return np.tile(A=self.parameters["support"], reps=size)
        elif isinstance(size, int):
            return np.tile(A=self.parameters["support"], reps=[size, *np.repeat(1, ndims)])
        else:
            return np.tile(A=self.parameters["support"], reps=tuple([*size, *np.repeat(1, ndims)]))

    def reshape(self, newshape):
        try:
            # Reshape support
            self._parameters["support"].reshape(newshape=newshape)
        except ValueError:
            raise ValueError("Cannot reshape this Dirac distribution to the given shape: {}".format(str(newshape)))

    # Binary arithmetic operations
    def __add__(self, other):
        if isinstance(other, Dirac):
            return Dirac(support=self.parameters["support"] + other.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__add__(other=self)

    def __sub__(self, other):
        if isinstance(other, Dirac):
            return Dirac(support=self.parameters["support"] - other.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__rsub__(other=self)

    def __mul__(self, other):
        if isinstance(other, Dirac):
            return Dirac(support=self.parameters["support"] * other.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__mul__(other=self)

    def __matmul__(self, other):
        if isinstance(other, Dirac):
            return Dirac(support=self.parameters["support"] @ other.parameters["support"],
                         random_state=self.random_state)
        else:
            return other.__rmatmul__(other=self)

    def __truediv__(self, other):
        if isinstance(other, Dirac):
            return Dirac(support=operator.truediv(self.parameters["support"], other.parameters["support"]),
                         random_state=self.random_state)
        else:
            return other.__rtruediv__(other=self)

    def __pow__(self, power, modulo=None):
        if isinstance(power, Dirac):
            return Dirac(support=pow(self.parameters["support"], power.parameters["support"], modulo),
                         random_state=self.random_state)
        else:
            return power.__rpow__(power=self, modulo=modulo)

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        return other.__add__(other=self)

    def __rsub__(self, other):
        return other.__sub__(other=self)

    def __rmul__(self, other):
        return other.__mul__(other=self)

    def __rmatmul__(self, other):
        return other.__matmul__(other=self)

    def __rtruediv__(self, other):
        return other.__truediv__(other=self)

    def __rpow__(self, power, modulo=None):
        return power.__pow__(power=self)

    # Augmented arithmetic assignments (+=, -=, *=, ...) attempting to do the operation in place
    def __iadd__(self, other):
        if isinstance(other, Dirac):
            self.parameters["support"] = self.parameters["support"] + other.parameters["support"]
            return self
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, Dirac):
            self.parameters["support"] = self.parameters["support"] - other.parameters["support"]
            return self
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, Dirac):
            self.parameters["support"] = self.parameters["support"] * other.parameters["support"]
            return self
        else:
            return NotImplemented

    def __imatmul__(self, other):
        if isinstance(other, Dirac):
            self.parameters["support"] = self.parameters["support"] @ other.parameters["support"]
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, Dirac):
            self.parameters["support"] = operator.truediv(self.parameters["support"], other.parameters["support"])
            return self
        else:
            return NotImplemented

    def __ipow__(self, power, modulo=None):
        if isinstance(power, Dirac):
            self.parameters["support"] = pow(self.parameters["support"], power.parameters["support"], modulo)
            return self
        else:
            return NotImplemented

    # Unary arithmetic operations
    def __neg__(self):
        self.parameters["support"] = operator.neg(self.parameters["support"])
        return self

    def __pos__(self):
        self.parameters["support"] = operator.pos(self.parameters["support"])
        return self

    def __abs__(self):
        self.parameters["support"] = operator.abs(self.parameters["support"])
        return self

    def __invert__(self):
        self.parameters["support"] = operator.invert(self.parameters["support"])
        return self
