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

    .. math:: \\dot y(t) = f(t, y(t)), \\quad t \\in [t_0, T].

    It provides options for defining custom right-hand side (RHS)
    functions, their Jacobians and closed form solutions.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    rhs : callable, signature: ``(t, y, **kwargs)``
        RHS function
        :math:`f : [t_0, T] \times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
        of the ODE system. As such it takes a float and an
        np.ndarray of shape (d,) and returns a np.ndarray
        of shape (d,). As of now, no vectorization is supported
        (nor needed).
    jac : callable, signature: ``(t, y, **kwargs)``, optional
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

    def __init__(self, timespan, rhs, jac=None, hess=None, sol=None):
        """
        Initialises basic ODE attributes.

        Essentially, all but the initial/boundary distribution
        which then determine the type of problem.

        We take timespan as a tuple to mimic the
        scipy.solve_ivp interface.
        """
        self._t0, self._tmax = timespan
        self._rhs = rhs
        self._jac = jac
        self._hess = hess
        self._sol = sol

    def __call__(self, t, y, **kwargs):
        """
        Piggybacks on self.rhs(t, y).
        """
        return self.rhs(t, y, **kwargs)

    def rhs(self, t, y, **kwargs):
        """
        Evaluates model function f.
        """
        return self._rhs(t, y, **kwargs)

    def jacobian(self, t, y, **kwargs):
        """
        Jacobian of model function f.
        """
        if self._jac is None:
            raise NotImplementedError
        else:
            return self._jac(t, y, **kwargs)

    def hessian(self, t, y, **kwargs):
        """
        Hessian of model function f.

        For :math:`d=3`, the Hessian
        :math:`H_f(t, y) \\in \\mathbb{R}^{3 \\times 3 \\times 3}`
        is expected be evaluated as

        .. math:: H_f(t, y) = \\left[H_{f_1}(t, y), H_{f_2}(t, y), H_{f_3}(t, y) \\right]^\\top

        since for any directions :math:`v_1, v_2` the outcome of
        :math:`H_f(t_0, y_0) \\cdot v_1 \\cdot v_2` is expected to contain
        the incline of :math:`f_i` in direction :math:`(v_1, v_2)`.
        """  # pylint: disable=line-too-long
        if self._hess is None:
            raise NotImplementedError
        else:
            return self._hess(t, y, **kwargs)

    def solution(self, t, **kwargs):
        """
        Solution of the IVP.
        """
        if self._sol is None:
            raise NotImplementedError
        else:
            return self._sol(t, **kwargs)

    @property
    def t0(self):
        return self._t0

    @property
    def tmax(self):
        return self._tmax

    @property
    def timespan(self):
        """
        Returns :math:`(t_0, T)` as ``[self.t0, self.tmax]``.
        Mainly here to provide an interface to scipy.integrate.
        Both :math:`t_0` and :math:`T` can be accessed via
        ``self.t0`` and ``self.tmax`` respectively.
        """
        return [self._t0, self._tmax]

    @property
    @abstractmethod
    def ndim(self):
        """
        Abstract, in order to force subclassing.
        """
        raise NotImplementedError
