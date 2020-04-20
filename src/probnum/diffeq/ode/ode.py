"""
Ordinary differential equations.

Subclassed by the types of ODEs: IVPs, BVPs and whatever you
can imagine.

See Also
--------
IVP : Initial value problems (ivp.py)
"""

from abc import ABC, abstractmethod


class ODE(ABC):
    """
    Ordinary differential equations.

    Extended by the types of ODEs, e.g. IVPs, BVPs.
    This class describes systems of irst order ordinary differential
    equations (ODEs),

    .. math:: \\dot x(t) = f(t, x(t)), \\quad t \\in [t_0, T].

    It provides options for defining custom right-hand side (RHS)
    functions, their Jacobians and closed form solutions.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    rhs : callable, signature: ``(t, x, **kwargs)``
        RHS function
        :math:`f : [t_0, T] \times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
        of the ODE system. As such it takes a float and an
        np.ndarray of shape (d,) and returns a np.ndarray
        of shape (d,). As of now, no vectorization is supported
        (nor needed).
    jac : callable, signature: ``(t, x, **kwargs)``, optional
        Jacobian of RHS function
        :math:`J_f : [0, T] \\times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
        of the ODE system. As such it takes a float and an
        np.ndarray of shape (d,) and returns a np.ndarray
        of shape (d,). As of now, no vectorization is supported
        (nor needed).
    sol : callable, signature: ``(t, **kwargs)``, optional
        Solution of the ODE system. Only well-defined in subclasses like
        IVP or BVP.

    See Also
    --------
    IVP : Extends ODE for initial value problems.
    """
    def __init__(self, timespan, rhs, jac=None, sol=None):
        """
        Initialises basic ODE attributes.

        Essentially, all but the initial/boundary distribution
        which then determine the type of problem.

        We take timespan as a tuple to mimic the
        scipy.solve_ivp interface.
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
        Returns :math:`(t_0, T)` as ``[self.t0, self.tmax]``.
        Mainly here to provide an interface to scipy.integrate.
        Both :math:`t_0` and :math:`T` can be accessed via
        ``self.t0`` and ``self.tmax`` respectively.
        """
        return [self.t0, self.tmax]

    @property
    @abstractmethod
    def ndim(self):
        """
        Abstract, in order to force subclassing.
        """
        raise NotImplementedError
