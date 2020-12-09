"""Definitions of problems solved by probabilistic numerical methods."""

import typing

import probnum.filtsmooth.statespace
import probnum.linops
import probnum.type


class RegressionProblem(typing.NamedTuple):
    r"""Regression problems.

    Fit a stochastic process to data, given a likelihood (realised by a :obj:`Transition`).
    Solved by, for instance, Kalman filtering and smoothing.
    """

    observations: np.ndarray
    locations: np.ndarray

    # Not sure how to best deal with this re. dependencies.
    likelihood: probnum.filtsmooth.statespace.DiscreteGaussian = None


class IVProblem(typing.NamedTuple):
    r"""First order ODE initial value problems.

    Compute a function :math:`y=y(t)` that solves

    .. math::
        \dot y(t) = f(t, y(t)), \quad y(0) = y_0

    Examples
    --------
    >>> ivp = IVProblem(lambda t, x: x*(1-x), 0., 3., 0.1)
    >>> ivp
    IVProblem(f=1.0, t0=1.0, tmax=1.0, y0=1.0)
    >>> ivp.f(ivp.t0, ivp.y0)
    0.09
    """

    f: typing.Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float  # Bold move: remove this? This is not really part of an IVP.
    y0: np.ndarray


class LinearSystemProblem(typing.NamedTuple):
    r"""Solve a linear system of equations: compute :math:`x` from :math;`Ax=b`."""

    A: Union[probnum.linops.LinearOperator, np.ndarray]
    b: np.ndarray  # Can linalg handle multiple RHSs?


class QuadratureProblem(typing.NamedTuple):
    r"""Numerically approximate an integral.

    Compute the integral

        .. math:
            \int_\Omega f(x) \diff \mu(x)

    for a function :math:`f: \Omega \rightarrow \mathbb{R}`.
    For the time being, :math:`\mu` is the Lebesgue measure.
    """

    integrand: typing.Callable[[np.ndarray], np.ndarray]
    domain: Callable[[np.ndarray], bool]  # Up for discussion...
