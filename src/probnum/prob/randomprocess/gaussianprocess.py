"""
Gaussian process interface.
"""

from probnum.prob.randomprocess.randomprocess import RandomProcess
from probnum.prob.randomprocess.covariance import *
from probnum.prob import Normal, RandomVariable


class GaussianProcess(RandomProcess):
    """
    We dont subclass from ContinuousProcess but from RandomProcess
    because we will not reuse any SDE machinery here.
    If you like to define a Gauss-Markov process use ContinuousProcess
    with a linear SDE and Gaussian initial variable
    """
    def __init__(self, meanfun=None, covfun=None, shape=None, dtype=None):
        """
        """
        if not issubclass(type(covfun), Covariance):
            covfun = Covariance(covfun=covfun)
        self._meanfun = meanfun
        self._covfun = covfun
        super().__init__(shape=shape, dtype=dtype)

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        if self._meanfun is not None and self._covfun is not None:
            dist = Normal(self.meanfun(x), self.covfun(x, x))
            return RandomVariable(distribution=dist)
        else:
            raise NotImplementedError("Mean and covariance of the "
                                      "GP are not implemented.")

    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def covfun(self, x1, x2):
        """
        Covariance (function) of the random process,
        also known as kernel.
        """
        raise NotImplementedError

    def sample(self, size=(), x=None, **kwargs):
        """
        Draw realizations from the random process.
        """
        if x is None:
            raise ValueError("Please specify a location x.")
        else:
            randvar = self.__call__(x)
            return randvar.sample(size=size)

    def condition(self, start, stop, randvar, **kwargs):
        """
        Conditions the random process on distribution randvar
        at time start. Returns RandomVariable representing its
        distribution at time stop.
        """
        raise NotImplementedError("Conditioning of the GP is not implemented.")

    def forward(self, start, stop, value, **kwargs):
        """
        Forwards a particle ``value`` according to the dynamics.
        Returns RandomVariable representing its
        distribution at time stop.

        This function allows using a random process like a transition
        density, sometimes without being one.
        """
        raise NotImplementedError("Forwarding the GP is not implemented.")
