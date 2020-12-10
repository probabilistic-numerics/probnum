"""Definitions of problems currently solved by probabilistic numerical methods.

If a problem is not listed here, we have not implemented a probabilistic
numerical method for it yet.
"""

import dataclasses
import typing

import numpy as np
import scipy.sparse

import probnum


@dataclasses.dataclass
class RegressionProblem:
    r"""Regression problem.

    Fit a stochastic process to data, given a likelihood (realised by a :obj:`Transition`).
    Solved by Kalman filtering and smoothing in :mod:`probnum.filtsmooth`.

    Parameters
    ----------
    observations
        Observations of the latent process.
    locations
        Grid-points on which the observations were taken.
    likelihood
        Likelihood of the observations; that is, relation between the latent process and the observed values.
        Encodes for example noise.

    Examples
    --------
    >>> obs = [11.4123, -15.5123]
    >>> loc = [0.1, 0.2]
    >>> rp = RegressionProblem(observations=obs, locations=loc)
    >>> rp
    RegressionProblem(observations=[11.4123, -15.5123], locations=[0.1, 0.2], likelihood=None)
    >>> rp.observations
    [11.4123, -15.5123]
    """

    observations: np.ndarray
    locations: np.ndarray

    # Optional, because it should be specifiable without explicit likelihood info.
    likelihood: probnum.filtsmooth.statespace.DiscreteGaussian = None


@dataclasses.dataclass
class IVProblem:
    r"""First order ODE initial value problem.

    Compute a function :math:`y=y(t)` that solves

    .. math::
        \dot y(t) = f(t, y(t)), \quad y(t_0) = y_0

    on time-interval :math:`[t_0, t_\text{max}]`.
    Solved by the ODE solvers in :mod:`probnum.diffeq`.


    Parameters
    ----------
    f
        ODE vector-field.
    t0
        Initial point in time.
    tmax
        Final point in time.
    y0
        Initial value of the solution.
    df
        Jacobian of the ODE vector-field :math:`f=f(t,y)` with respect to the :math:`y` variable.
    ddf
        Hessian of the ODE vector-field :math:`f=f(t,y)` with respect to the :math:`y` variable.

    Examples
    --------
    >>> def f(t, x):
    ...     return x*(1-x)
    >>> ivp = IVProblem(f, t0=0., tmax=3., y0=0.1)
    >>> ivp.t0, ivp.tmax, ivp.y0
    (0.0, 3.0, 0.1)
    >>> np.round(ivp.f(ivp.t0, ivp.y0), 2)
    0.09
    """

    f: typing.Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float  # Bold move: remove this? This is not really part of an IVP.
    y0: typing.Union[probnum.type.FloatArgType, np.ndarray]
    df: typing.Callable[[float, np.ndarray], np.ndarray] = None
    ddf: typing.Callable[[float, np.ndarray], np.ndarray] = None


@dataclasses.dataclass
class LinearSystemProblem:
    r"""Linear system of equations.

    Compute :math:`x` from :math:`Ax=b`.
    Solved by the probabilistic linear solver in mod:`probnum.linalg`

    Parameters
    ----------
    A
        Square matrix or linear operator.
    b
        Right-hand side vector or matrix.

    Examples
    --------
    >>> A = np.eye(3)
    >>> b = np.arange(3)
    >>> lsp = LinearSystemProblem(A, b)
    >>> lsp
    LinearSystemProblem(A=array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), b=array([0, 1, 2]))
    """

    A: typing.Union[
        np.ndarray,
        "scipy.sparse.spmatrix",
        "probnum.linops.LinearOperator",
        "probnum.RandomVariable",
    ]
    b: typing.Union[np.ndarray, "probnum.RandomVariable"]


@dataclasses.dataclass
class QuadProblem:
    r"""Numerical computation of an integral.

    Compute the integral

        .. math::
            \int_\Omega f(x) \, \text{d} \mu(x)

    for a function :math:`f: \Omega \rightarrow \mathbb{R}`.
    For the time being, :math:`\mu` is the Lebesgue measure.
    Solved by the quadrature rules in :mod:`probnum.quad`.

    Parameters
    ----------
    integrand
        Function to be integrated.
    domain
        Domain of the integral.

    Examples
    --------
    >>> def integrand(x):
    ...     return x**2
    >>> def domain(x):
    ...     return 0 < x < 1
    >>> qp = QuadProblem(integrand, domain)
    >>> np.round(qp.integrand(0.2), 2)
    0.04
    >>> qp.domain(0.2)
    True
    """

    integrand: typing.Callable[[np.ndarray], np.ndarray]
    domain: typing.Callable[[np.ndarray], bool]  # Up for discussion...
