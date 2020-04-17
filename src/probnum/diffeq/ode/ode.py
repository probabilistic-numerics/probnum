"""
Ordinary differential equations.

Subclassed by the types of ODEs: IVPs, BVPs and whatever you
can imagine.

See Also
--------
IVP : Initial value problems (ivp.py)
"""

from abc import ABC, abstractmethod


__all__ = ["ODE"]

class ODE(ABC):
    """
    Base Ordinary differential equation class.

    Subclassed by the types of ODEs: IVPs, BVPs and whatever you
    can imagine.
    """
    def __init__(self, timespan, rhs, jac=None, sol=None):
        """
        Initialises basic ODE attributes.

        Essentially, all but the initial/boundary distribution
        which then determine the type of problem.

        We take timespan as a tuple to mimic the
        scipy.solve_ivp interface.

        timespan : (t0, tmax); tuple of two floats,
            time span of IVP
        initdist : randomvariable.RandomVariable,
            usually dirac.Dirac (noise-free)
            or gaussian.MultivariateGaussian (noisy)
        rhs : callable, signature (t, x)
            right hand side vector field function
        jac : callable, signature (t, x)
            Jacobian of right hand side function
        """
        self.t0, self.tmax = timespan
        self.rhs = rhs
        self.jac = jac
        self.sol = sol

    def rhs(self, t, x, **kwargs):
        """
        Evaluates model function f.
        """
        return self.rhs(t, x, **kwargs)

    def jacobian(self, t, x, **kwargs):
        """
        Jacobian of model function f.
        """
        if self.jac is None:
            raise NotImplementedError
        else:
            return self.jac(t, x, **kwargs)

    def solution(self, t, **kwargs):
        """
        Solution of the IVP.
        """
        if self.sol is None:
            raise NotImplementedError
        else:
            return self.sol(t, **kwargs)

    @property
    def timespan(self):
        """
        """
        return [self.t0, self.tmax]

    @property
    @abstractmethod
    def ndim(self):
        """
        Abstract, in order to force subclassing.
        """
        raise NotImplementedError
