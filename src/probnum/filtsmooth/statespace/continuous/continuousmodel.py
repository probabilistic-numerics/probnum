"""
Continuous Markov models implicitly defined through SDEs,
dx(t) = f(t, x(t)) dt + l(t, x(t)) dB(t).
Possible input for StateSpace.dynmod or StateSpace.measmod.

Note
----
ContinuousModel only implements
StateSpaceComponent.sample(). It passes on the
responsibility of implementing chapmankolmogorov() to its
subclasses and creates further abstract methods drift(),
dispersion() as well as a hook for jacobian().

When implementing continuous models we think of
Gauss-Markov models (see linearsde.py)
but do our very best to not enforce Gaussianity.
In the future, this might be more general than Gauss-
Markov models.
"""

from abc import ABC, abstractmethod

import numpy as np


__all__ = ["ContinuousModel"]


class ContinuousModel(ABC):
    """
    Interface for time-continuous
    Markov models of the form
    dx = f(t, x) dt + l(t, x) dBt.

    In the language of dynamic models:
    x(t) : state process
    f(t, x(t)) : drift function
    l(t, x(t)) : dispersion matrix.
    B(t) : Brownian motion with const. diffusion matrix Q.
    """

    def sample(self, start, stop, step, initstate, **kwargs):
        """
        Samples from ``initstate`` at ``start`` to ``stop`` with
        stepsize ``step``.

        Start, stop and step lead to a np.arange-like
        interface.
        Returns a single element at the end of the time, not the
        entire array!
        """
        if not isinstance(initstate, np.ndarray):
            raise ValueError("Init state is not array!")
        times = np.arange(start, stop, step)
        currstate = initstate
        for idx in range(1, len(times)):
            bmsamp = np.random.multivariate_normal(
                np.zeros(len(self.diffusionmatrix)), self.diffusionmatrix * step
            )
            driftvl = self.drift(times[idx - 1], currstate, **kwargs)
            dispvl = self.dispersion(times[idx - 1], currstate, **kwargs)
            currstate = currstate + step * driftvl + dispvl @ bmsamp
        return currstate

    @abstractmethod
    def drift(self, time, state, **kwargs):
        """
        Evaluates f = f(t, x(t)).
        """
        raise NotImplementedError

    @abstractmethod
    def dispersion(self, time, state, **kwargs):
        """
        Evaluates l = l(t, x(t)).
        """
        raise NotImplementedError

    def jacobian(self, time, state, **kwargs):
        """
        Jacobian of drift w.r.t. state: d_x f(t, x(t))
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def diffusionmatrix(self):
        """
        Evaluates Q.
        In 1D, this is \\sigma^2.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ndim(self):
        """
        Spatial dimension (utility attribute).
        """
        raise NotImplementedError

    def chapmankolmogorov(self, start, stop, step, randvar, **kwargs):
        """
        If available, this returns the closed form solution to the
        Chapman-Kolmogorov equations (CKEs).

        Solutions to the CKEs are important in filtering.

        Available for instance for linear SDEs.
        """
        raise NotImplementedError("Chap.-Kolg. not implemented.")
