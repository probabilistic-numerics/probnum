"""Definitions of problems currently solved by probabilistic numerical methods."""

import dataclasses
import typing

import numpy as np

import probnum.filtsmooth.statespace
import probnum.linops
import probnum.type


@dataclasses.dataclass
class RegressionProblem:
    r"""Regression problems.

    Fit a stochastic process to data, given a likelihood (realised by a :obj:`Transition`).
    Solved by Kalman filtering and smoothing in :mod:`probnum.filtsmooth`.
    """

    observations: np.ndarray
    locations: np.ndarray

    # Not sure how to best deal with this re. dependencies.
    likelihood: probnum.filtsmooth.statespace.DiscreteGaussian = None


@dataclasses.dataclass
class IVProblem:
    r"""First order ODE initial value problems.

    Compute a function :math:`y=y(t)` that solves

    .. math::
        \dot y(t) = f(t, y(t)), \quad y(0) = y_0

    Solved by the ODE solvers in :mod:`probnum.diffeq`.

    Examples
    --------
    >>> def f(t, x):
    ...     return x*(1-x)
    >>> ivp = IVProblem(f, 0., 3., 0.1)
    >>> ivp.t0, ivp.tmax, ivp.y0
    (0.0, 3.0, 0.1)
    >>> np.round(ivp.f(ivp.t0, ivp.y0), 2)
    0.09
    """

    f: typing.Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float  # Bold move: remove this? This is not really part of an IVP.
    y0: np.ndarray


@dataclasses.dataclass
class LinearSystemProblem:
    r"""Solve a linear system of equations.

    Compute :math:`x` from :math:`Ax=b`.
    Solved by the probabilistic linear solver in mod:`probnum.linalg`

    Example
    -------
    >>> A = np.eye(3)
    >>> b = np.arange(3)
    >>> lsp = LinearSystemProblem(A, b)
    >>> lsp
    LinearSystemProblem(A=array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), b=array([0, 1, 2]))
    """

    A: typing.Union[probnum.linops.LinearOperator, np.ndarray]
    b: np.ndarray  # Can probnum.linalg handle multiple RHSs?


@dataclasses.dataclass
class QuadProblem:
    r"""Numerically approximate an integral.

    Compute the integral

        .. math::
            \int_\Omega f(x) \, \text{d} \mu(x)

    for a function :math:`f: \Omega \rightarrow \mathbb{R}`.
    For the time being, :math:`\mu` is the Lebesgue measure.
    Solved by the quadrature rules in :mod:`probnum.quad`.

    Example
    -------
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
