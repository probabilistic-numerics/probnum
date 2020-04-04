"""
Normal distribution base class.

Each of type of normal distribution inherits from this base class.

It is internal. For public use, refer to normal.Normal instead.
"""


import operator
from probnum.prob.distributions.distribution import Distribution
from probnum.prob.distributions.dirac import Dirac

class _Normal(Distribution):
    """
    Base class for normal distributions.

    Contains initialisation as a Distribution object plus essential
    methods (properties & arithmetics).
    """

    def __init__(self, mean=0., cov=1., random_state=None):
        super().__init__(parameters={"mean": mean, "cov": cov}, dtype=float,
                         random_state=random_state)

    def mean(self):
        return self.parameters["mean"]

    def cov(self):
        return self.parameters["cov"]

    def var(self):
        raise NotImplementedError

    # Binary arithmetic operations #####################################

    def __add__(self, other):
        """
        Only works if the other dist. is a Dirac.

        The type(self)(...) makes sure that adding a Dirac to, say,
        a _MultivariateNormal returns a _MultivariateNormal.
        """
        if isinstance(other, Dirac):
            return type(self)(mean=self.mean() + other.mean(),
                              cov=self.cov(),
                              random_state=self.random_state)
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return self + (-other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Only works if the other dist. is a Dirac.

        The type(self)(...) makes sure that adding a Dirac to, say,
        a _MultivariateNormal returns a _MultivariateNormal.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            if delta == 0:
                return Dirac(support=0 * self.mean(),
                             random_state=self.random_state)
            else:
                return type(self)(mean=self.mean() * delta,
                                  cov=self.cov() * delta ** 2,
                                  random_state=self.random_state)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if other == 0:
            raise ZeroDivisionError("Division by zero not supported.")
        else:
            if isinstance(other, Dirac):
                return self * operator.inv(other)
            else:
                return NotImplemented

    def __pow__(self, power, modulo=None):
        return NotImplemented

    # Binary arithmetic operations with reflected operands #############

    def __radd__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return self + other
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return operator.neg(self) + other
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return self * other
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        Only works if the other dist. is a Dirac.

        The type(self)(...) makes sure that adding a Dirac to, say,
        a _MultivariateNormal returns a _MultivariateNormal.
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            return type(self)(mean=delta @ self.mean(),
                              cov=delta @ (self.cov() @ delta.transpose()),
                              random_state=self.random_state)
        return NotImplemented

    def __rtruediv__(self, other):
        """
        Only works if the other dist. is a Dirac.
        """
        if isinstance(other, Dirac):
            return operator.inv(self) * other
        else:
            return NotImplemented

    def __rpow__(self, power, modulo=None):
        return NotImplemented

    # # Augmented arithmetic assignments (+=, -=, *=, ...) ###############
    # # attempting to do the operation in place ##########################
    #
    # def __iadd__(self, other):
    #     return NotImplemented
    #
    # def __isub__(self, other):
    #     return NotImplemented
    #
    # def __imul__(self, other):
    #     return NotImplemented
    #
    # def __imatmul__(self, other):
    #     return NotImplemented
    #
    # def __itruediv__(self, other):
    #     return NotImplemented
    #
    # def __ipow__(self, power, modulo=None):
    #     return NotImplemented

    # Unary arithmetic operations ######################################

    def __neg__(self):
        """
        Negation of r.v.
        """
        try:
            return _Normal(mean=- self.mean(),
                           cov=self.cov(),
                           random_state=self.random_state)
        except Exception:
            return NotImplemented

    def __pos__(self):
        try:
            return _Normal(mean=operator.pos(self.mean()),
                           cov=self.cov(),
                           random_state=self.random_state)
        except Exception:
            return NotImplemented

    def __abs__(self):
        try:
            # todo: add absolute moments of normal (see: https://arxiv.org/pdf/1209.4340.pdf)
            return Distribution(parameters={},
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            return NotImplemented

    def __invert__(self):
        try:
            return Distribution(parameters={},
                                sample=lambda size: operator.abs(self.sample(size=size)),
                                random_state=self.random_state)
        except Exception:
            return NotImplemented

