"""The folder is called "ode" but this module is "ivp" because in the future, there
might be more ode-based problems, such as bvp."""
# pylint: disable=unused-variable

import numpy as np

from probnum.diffeq.ode.ode import ODE


class IVP(ODE):
    """Initial value problems (IVP).

    This class descibes initial value problems based on systems of
    first order ordinary differential equations (ODEs),

    .. math:: \\dot y(t) = f(t, y(t)), \\quad y(t_0) = y_0,
        \\quad t \\in [t_0, T]

    It provides options for defining custom right-hand side (RHS)
    functions, their Jacobians and closed form solutions.

    Since we use them for probabilistic ODE solvers these functions
    fit into the probabilistic framework as well. That is,
    the initial value is a RandomVariable object with some
    distribution that reflects the prior belief over the initial
    value. To recover "classical" initial values one can use a
    Constant random variable.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Constant (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use a :class:`~probnum.random_variables.Constant` random variable.
        Implementation depends on the mean of this RandomVariable,
        so please only use RandomVariable objects with available
        means, e.g. Constants or Normals.
    rhs : callable, signature: ``(t, y, **kwargs)``
        RHS function
        :math:`f : [0, T] \\times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
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
        Solution of IVP.

    See Also
    --------
    ODE : Abstract interface for  ordinary differential equations.

    Examples
    --------
    >>> from probnum.diffeq import IVP
    >>> rhsfun = lambda t, y, **kwargs: 2.0*y
    >>> from probnum import randvars
    >>> initrv = randvars.Constant(0.1)
    >>> timespan = (0, 10)
    >>> ivp = IVP(timespan, initrv, rhsfun)
    >>> print(ivp.rhs(0., 2.))
    4.0
    >>> print(ivp.timespan)
    [0, 10]
    >>> print(ivp.t0)
    0

    >>> initrv = randvars.Normal(0.1, 1.0)
    >>> ivp = IVP(timespan, initrv, rhsfun)
    >>> jac = lambda t, y, **kwargs: 2.0
    >>> ivp = IVP(timespan, initrv, rhs=rhsfun, jac=jac)
    >>> print(ivp.rhs(0., 2.))
    4.0
    >>> print(ivp.jacobian(100., -1))
    2.0
    """

    def __init__(self, timespan, initrv, rhs, jac=None, hess=None, sol=None):

        self.initrv = initrv
        super().__init__(timespan=timespan, rhs=rhs, jac=jac, hess=hess, sol=sol)

    @property
    def initialdistribution(self):
        """Distribution of the initial random variable."""
        return self.initrv

    @property
    def initialrandomvariable(self):
        """Initial random variable."""
        return self.initrv

    @property
    def dimension(self):
        """Spatial dimension of the IVP problem.

        Depends on the mean of the initial random variable.
        """
        if np.isscalar(self.initrv.mean):
            return 1
        else:
            return len(self.initrv.mean)
