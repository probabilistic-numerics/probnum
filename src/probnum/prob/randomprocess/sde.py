"""
Stochastic differential equations.
"""
import numpy as np
from probnum.prob.distributions import *
from probnum.prob.randomprocess.transitions import Transition


class SDE(Transition):
    """
    Stochastic differential equations.

    Represents stochastic differential equations (SDEs) of the form

    .. math:: dX(t) = f(t, X(t)) dt + l(t, X(t)) dB(t),
        \\quad X(0) \\sim p(X(0))

    driven by Brownian motion.

    Inherits from Transition to provide the same interface.
    In particular, SDEs need the interfaces forward() and condition().
    (In some sense, SDEs are just continuous-time transition models
    anyway...)


    Parameters
    ----------
    driftfun : callable, signature=``(t, x)``

    dispfun : callable, signature=``(t, x)``

    difffun : callable, signature=``(t)``

    initrv : RandomVariable

    jacobfun, callable, signature=``(t, x)``, optional.
        Jacobian of the drift function.
        Only the derivatives w.r.t. the spatial variable x.
    """
    def __init__(self, driftfun, dispfun, difffun, jacobfun=None):
        """ """
        self._driftfun = driftfun
        self._dispfun = dispfun
        self._difffun = difffun
        self._jacobfun = jacobfun

    def driftfun(self, t, x):
        """ """
        return self._driftfun(t, x)

    def dispfun(self, t, x, **kwargs):
        """ """
        return self._driftfun(t, x)

    def diffmatfun(self, t, x):
        """ """
        return self._driftfun(t, x)

    def jacobfun(self, t, x):
        """ """
        if self.jacobfun is not None:
            return self._jacobfun(t, x)
        else:
            return NotImplementedError

    def simulate(self, times, size=(), nsteps=1):
        """
        Numerically simulates N trajectories of the solution.

        Uses an Euler-Maruyama scheme and can be used to sample from
        a random process that is defined through solving the SDE.
        """
        raise NotImplementedError("This should be possible (todo?)")

    def solve(self, start, stop, step, randvar):
        """
        Numerically solves SDE on interval [start, stop] with step
        size step starting with initial distribution randvar.

        Returns the entire distribution, i.e. an array of
        RandomVariable objects.
        """
        raise NotImplementedError

    def condition(self, start, stop, randvar, **kwargs):
        """
        Numerically solves SDE on interval [start, stop] with step
        size step starting with initial distribution randvar.
        """
        raise NotImplementedError

    def forward(self, start, stop, value, **kwargs):
        """
        Numerically approximates the distribution of the SDE process
        at time stop conditioned on the fact that randvar hits a
        prescribed value at time start.

        Returns only the final distribution, i.e. a single
        RandomVariable object.
        """
        raise NotImplementedError


class LinearSDE(SDE):
    """
    Linear stochastic differential equations.

    Represents linear stochastic differential equations (SDEs) of the
    form

    .. math:: dX(t) = [F(t) X(t) + u(t)]dt + L(t) X(t) dB(t),
        \\quad X(0) \\sim p(X(0))

    driven by Brownian motion. That is, drift and dispersion are linear
    in the space-variable.

    If the initial distribution is a Normal distribution, solutions to
    this type of process are Gauss-Markov processes.
    """
    def __init__(self, driftfun, forcefun, dispfun, difffun):
        """ """
        if np.isscalar(initrv.sample()):
            super().__init__(driftfun=(lambda t, x: driftfun(t, x) * x + forcefun(t)),
                             dispfun=(lambda t, x: dispfun(t, x) * x),
                             difffun=difffun)
        else:
            super().__init__(driftfun=(lambda t, x: driftfun(t, x) @ x),
                             dispfun=(lambda t, x: dispfun(t, x) @ x),
                             difffun=difffun,
                             jacobfun=driftfun)

    def solve(self, start, stop, step, randvar):
        """
        Numerically solves SDE on interval [start, stop] with step
        size step starting with initial distribution randvar.

        Returns the entire distribution, i.e. an array of
        RandomVariable objects.
        """
        raise NotImplementedError

    def condition(self, start, stop, randvar, **kwargs):
        """
        Numerically solves SDE on interval [start, stop] with step
        size step starting with initial distribution randvar.
        """
        raise NotImplementedError

    def forward(self, start, stop, value, **kwargs):
        """
        Numerically approximates the distribution of the SDE process
        at time stop conditioned on the fact that randvar hits a
        prescribed value at time start.

        Returns only the final distribution, i.e. a single
        RandomVariable object.
        """
        raise NotImplementedError


class LTISDE(LinearSDE):
    """
    Linear time-invariant stochastic differential equations.

    Represents linear time-invariant stochastic differential equations
    (SDEs) of the form

    .. math:: dX(t) = [F X(t) + u]dt + L X(t) dB(t),
        \\quad X(0) \\sim p(X(0))

    driven by Brownian motion with constant diffusion.
    That is, drift, force, dispersion and diffusion are all linear
    in the space-variable and time-invariant.
    """
    def __init__(self, driftmat, forcevec, dispmat, diffmat):
            pass
            # normal init

    def solve(self, start, stop, step, randvar):
        """
        Numerically solves SDE on interval [start, stop] with step
        size step starting with initial distribution randvar.

        Returns the entire distribution, i.e. an array of
        RandomVariable objects.
        """
        raise NotImplementedError

    def condition(self, start, stop, randvar, **kwargs):
        """
        Numerically solves SDE on interval [start, stop] with step
        size step starting with initial distribution randvar.
        """
        raise NotImplementedError

    def forward(self, start, stop, value, **kwargs):
        """
        Numerically approximates the distribution of the SDE process
        at time stop conditioned on the fact that randvar hits a
        prescribed value at time start.

        Returns only the final distribution, i.e. a single
        RandomVariable object.
        """
        raise NotImplementedError
