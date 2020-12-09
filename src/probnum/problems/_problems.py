"""Definitions of problems solved by probabilistic numerical methods."""

import typing

import probnum.filtsmooth.statespace
import probnum.linops
import probnum.type


class RegressionProblem(typing.NamedTuple):
    r"""Regression problems.

    Find a stochastic process that fits data given a likelihood (realised by a :obj:`Transition`). Solved by, for instance, Kalman filtering and smoothing.
    """

    observations: np.ndarray
    locations: np.ndarray
    likelihood: probnum.filtsmooth.statespace.DiscreteGaussian  # Not sure how to deal with this.


class IVProblem(typing.NamedTuple):
    r"""ODE initial value problems.

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
    tmax: float
    y0: np.ndarray  # Formerly (and unnecessarily) a RandomVariable


class LinearSystemProblem(typing.NamedTuple):
    r"""Solve a linear system of equations: compute :math:`x` from :math;`Ax=b`."""

    A: Union[probnum.linops.LinearOperator, np.ndarray]
    b: np.ndarray  # Can linalg handle multiple RHSs?


class QuadratureProblem(typing.NamedTuple):
    r"""Numerically approximate an integral.

    Compute :math:`\int_\Omega f(x) \diff \mu(x)` for a function :math:`f: \Omega \rightarrow \mathbb{R}`. For the time being, :math:`\mu` is the Lebesgue measure.
    """

    integrand: typing.Callable[[np.ndarray], np.ndarray]
    domain: Callable[[np.ndarray], bool]  # Up for discussion...
