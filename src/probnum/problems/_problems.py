"""Definitions of problems currently solved by probabilistic numerical methods."""

import dataclasses
import typing

import numpy as np
import scipy.sparse

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum  # pylint: disable="unused-import"
import probnum.filtsmooth as pnfs
import probnum.linops as linops
import probnum.random_variables as rvs
import probnum.type as pntp
from probnum.utils import as_random_state

# pylint: disable="invalid-name"


@dataclasses.dataclass
class RegressionProblem:
    r"""Regression problem.

    Fit a stochastic process to data, given a likelihood (realised by a
    :obj:`DiscreteGaussian` transition). Solved by filters and smoothers in
    :mod:`probnum.filtsmooth`.

    Parameters
    ----------
    observations
        Observations of the latent process.
    locations
        Grid-points on which the observations were taken.
    likelihood
        Likelihood of the observations; that is, relation between the latent process and
        the observed values. Encodes for example noise.
    solution
        Closed form, analytic solution to the problem. Used for testing and benchmarking.

    Examples
    --------
    >>> obs = [11.4123, -15.5123]
    >>> loc = [0.1, 0.2]
    >>> rp = RegressionProblem(observations=obs, locations=loc)
    >>> rp
    RegressionProblem(observations=[11.4123, -15.5123], locations=[0.1, 0.2], likelihood=None, solution=None)
    >>> rp.observation
    [11.4123, -15.5123]
    """

    observations: np.ndarray
    locations: np.ndarray

    # Optional, because it should be specifiable without explicit likelihood info.
    # 'DiscreteGaussian' is currently in 'statespace', but can be used to define general
    # Likelihood functions; see #282
    likelihood: typing.Optional[pnfs.statespace.DiscreteGaussian] = None

    # For testing and benchmarking
    solution: typing.Optional[
        typing.Callable[[np.ndarray], typing.Union[float, np.ndarray]]
    ] = None


@dataclasses.dataclass
class InitialValueProblem:
    r"""First order ODE initial value problem.

    Compute a function :math:`y=y(t)` that solves

    .. math::
        \dot y(t) = f(t, y(t)), \quad y(t_0) = y_0

    on time-interval :math:`[t_0, t_\text{max}]`.
    Solved by probabilistic ODE solvers in :mod:`probnum.diffeq`.


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
        Jacobian of the ODE vector-field :math:`f=f(t,y)` with respect to the :math:`y`
        variable.
    ddf
        Hessian of the ODE vector-field :math:`f=f(t,y)` with respect to the :math:`y`
        variable.
    solution
        Closed form, analytic solution to the problem. Used for testing and
        benchmarking.

    Examples
    --------
    >>> import numpy as np
    >>> def f(t, x):
    ...     return x * (1 - x)
    >>> ivp = InitialValueProblem(f, t0=0., tmax=3., y0=0.1)
    >>> ivp.t0, ivp.tmax, ivp.y0
    (0.0, 3.0, 0.1)
    >>> np.round(ivp.f(ivp.t0, ivp.y0), 2)
    0.09
    """

    f: typing.Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float
    y0: typing.Union[pntp.FloatArgType, np.ndarray]
    df: typing.Optional[typing.Callable[[float, np.ndarray], np.ndarray]] = None
    ddf: typing.Optional[typing.Callable[[float, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: typing.Optional[typing.Callable[[float, np.ndarray], np.ndarray]] = None


@dataclasses.dataclass(frozen=True)
class LinearSystem:
    r"""Linear system of equations.

    Compute :math:`x` from :math:`Ax=b`.
    Solved by probabilistic linear solvers in :mod:`probnum.linalg`

    Parameters
    ----------
    A
        System matrix or linear operator.
    b
        Right-hand side vector or matrix.
    solution
        True solution to the problem. Used for testing and benchmarking.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.eye(3)
    >>> b = np.arange(3)
    >>> lin_sys = LinearSystem(A, b)
    >>> lin_sys
    LinearSystem(A=array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), b=array([[0],
           [1],
           [2]]), solution=None)
    """

    A: typing.Union[
        np.ndarray,
        scipy.sparse.spmatrix,
        linops.LinearOperator,
        rvs.RandomVariable,
    ]
    b: typing.Union[np.ndarray, rvs.RandomVariable]

    # For testing and benchmarking
    solution: typing.Optional[typing.Union[np.ndarray, rvs.RandomVariable]] = None

    def __post_init__(self):
        # Check types
        linop_types = (
            np.ndarray,
            scipy.sparse.spmatrix,
            scipy.sparse.linalg.LinearOperator,
            rvs.RandomVariable,
        )
        vector_types = (np.ndarray, scipy.sparse.spmatrix, rvs.RandomVariable)
        if not isinstance(self.A, linop_types):
            raise TypeError(
                "A must be either an array, a linear operator or a random variable."
            )
        if not isinstance(self.b, vector_types):
            raise TypeError(
                "The right hand side must be a (sparse) array or a random variable."
            )
        if self.solution is not None:
            if not isinstance(self.solution, vector_types):
                raise TypeError(
                    "The solution must be a (sparse) array or a random variable."
                )

        # Check and normalize shapes
        if self.b.ndim == 1:
            # Reshape immutable object
            object.__setattr__(self, "b", self.b.reshape((-1, 1)))
        if self.solution is not None:
            if self.solution.ndim == 1:
                # Reshape immutable object
                object.__setattr__(self, "solution", self.solution.reshape((-1, 1)))
            if self.solution.ndim != 2:
                raise ValueError("Solution must be two-dimensional.")
        if self.A.ndim != 2 or self.b.ndim != 2:
            raise ValueError("System components must be two-dimensional.")

        # Check shape mismatch
        def dim_mismatch_error(arg0, arg1, arg0_name, arg1_name):
            return ValueError(
                f"Dimension mismatch. The shapes of {arg0_name} : {arg0.shape} "
                f"and {arg1_name} : {arg1.shape} must match."
            )

        if self.A.shape[0] != self.b.shape[0]:
            raise dim_mismatch_error(self.A, self.b, "A", "b")

        if self.solution is not None:
            if self.A.shape[1] != self.solution.shape[0]:
                raise dim_mismatch_error(self.A, self.solution, "A", "x")

            if self.solution.shape[1] != self.b.shape[1]:
                raise dim_mismatch_error(self.solution, self.b, "x", "b")

    @classmethod
    def from_matrix(cls, A, random_state=None) -> "probnum.problems.LinearSystem":
        """Generate a random linear system from a given (fixed) matrix or linear
        operator.

        Parameters
        ----------
        A :
            System matrix for the random linear system.
        random_state
            Random state of the random variable. If None (or np.random), the global
            :mod:`numpy.random` state is used. If integer, it is used to seed the local
            :class:`~numpy.random.RandomState` instance.
        """
        rng = as_random_state(random_state)
        solution = rng.normal(size=(A.shape[1], 1))
        right_hand_side = A @ solution

        return LinearSystem(A=A, solution=solution, b=right_hand_side)

    @cached_property
    def shape(self) -> pntp.ShapeType:
        """Shape of the linear system.

        Defined as the shape of the system matrix :code:`(m, n)` and the
        number of right hand sides :code:`(nrhs)`.
        """
        return self.A.shape + (self.b.shape[1],)


@dataclasses.dataclass
class QuadratureProblem:
    r"""Numerical computation of an integral.

    Compute the integral

        .. math::
            \int_\Omega f(x) \, \text{d} \mu(x)

    for a function :math:`f: \Omega \rightarrow \mathbb{R}`.
    For the time being, :math:`\mu` is the Lebesgue measure.
    Solved by quadrature rules in :mod:`probnum.quad`.

    Parameters
    ----------
    integrand
        Function to be integrated.
    lower_bd
        A number or a vector representing the lower bounds of the integral.
    upper_bd
        A number or a vector representing the upper bounds of the integral.
    output_dim
        Output dimension of the integrand.
    solution
        Closed form, analytic solution to the problem. Used for testing and
        benchmarking.

    Examples
    --------
    >>> import numpy as np
    >>> def integrand(x):
    ...     return np.linalg.norm(x)**2
    >>> lower_bd = 0.41
    >>> upper_bd = 4.32
    >>> qp1d = QuadratureProblem(integrand, lower_bd=lower_bd, upper_bd=upper_bd)
    >>> np.round(qp1d.integrand(0.2), 2)
    0.04
    >>> qp1d.lower_bd
    0.41
    >>> lower_bd = [0., 0.]
    >>> upper_bd = [1., 1.]
    >>> qp2d = QuadratureProblem(integrand, lower_bd=lower_bd, upper_bd=upper_bd)
    >>> qp2d.upper_bd
    [1.0, 1.0]
    """

    integrand: typing.Callable[[np.ndarray], typing.Union[float, np.ndarray]]
    lower_bd: typing.Union[pntp.FloatArgType, np.ndarray]
    upper_bd: typing.Union[pntp.FloatArgType, np.ndarray]
    output_dim: typing.Optional[int] = 1

    # For testing and benchmarking
    solution: typing.Optional[
        typing.Union[float, np.ndarray, rvs.RandomVariable]
    ] = None
