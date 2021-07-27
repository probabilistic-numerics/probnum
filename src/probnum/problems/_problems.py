"""Definitions of problems currently solved by probabilistic numerical methods."""

import dataclasses
from collections import abc
from typing import Callable, Optional, Sequence, Union

import numpy as np
import scipy.sparse

from probnum import linops, randvars
from probnum.typing import FloatArgType


@dataclasses.dataclass
class TimeSeriesRegressionProblem:
    r"""Time series regression problem.

    Fit a stochastic process to data, given a likelihood (realised by a :obj:`DiscreteGaussian` transition).
    Solved by filters and smoothers in :mod:`probnum.filtsmooth`.

    Parameters
    ----------
    observations
        Observations of the latent process.
    locations
        Grid-points on which the observations were taken.
    measurement_models
        Measurement models.
    solution
        Array containing solution to the problem at ``locations``. Used for testing and benchmarking.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum import randprocs
    >>> obs = [11.4123, -15.5123]
    >>> loc = [0.1, 0.2]
    >>> model = randprocs.markov.discrete.DiscreteLTIGaussian(np.ones((1, 1)), np.ones(1), np.ones((1,1)))
    >>> measurement_models = [model, model]
    >>> rp = TimeSeriesRegressionProblem(observations=obs, locations=loc, measurement_models=measurement_models)
    >>> rp
    TimeSeriesRegressionProblem(locations=[0.1, 0.2], observations=[11.4123, -15.5123], measurement_models=[DiscreteLTIGaussian(input_dim=1, output_dim=1), DiscreteLTIGaussian(input_dim=1, output_dim=1)], solution=None)
    >>> rp.observations
    [11.4123, -15.5123]

    Regression problems are also indexable.

    >>> len(rp)
    2
    >>> rp[0]
    (0.1, 11.4123, DiscreteLTIGaussian(input_dim=1, output_dim=1))
    """

    # The types are 'Sequence' (e.g. lists, tuples) or 'ndarray',
    # because we need __len__ and __getitem__, which both provide.
    # 'ndarray's are not 'Sequence's: https://github.com/numpy/numpy/issues/2776
    locations: Union[Sequence, np.ndarray]
    observations: Union[Sequence, np.ndarray]
    measurement_models: Union[Sequence, np.ndarray]

    # For testing and benchmarking
    # The solution here is a Sequence or array of the values of the truth at the location.
    solution: Optional[Union[Sequence, np.ndarray]] = None

    def __post_init__(self):
        """Some postprocessing of the time-series regression problem inits.

        1. Wrap the measurement models into an iterable of measurement models (by
        default, a list).
        2. Check that all inputs have the same length.
        """

        # If a single measurement model has been provided, transform it into a list of models
        # to ensure that there is a measurement model for every (t, y) combination.
        if not isinstance(self.measurement_models, abc.Iterable):
            self.measurement_models = [self.measurement_models] * len(self.locations)

        # Check that the lengths align. Uneven lengths are not supported
        # at the moment, because it is not clear how these should be handled.
        lengths_equal = (
            len(self.observations)
            == len(self.locations)
            == len(self.measurement_models)
        )
        if not lengths_equal:
            errormsg = "Lengths of the inputs do not match. "
            len_obs = f"len(observations)={len(self.observations)}. "
            len_loc = f"len(locations)={len(self.locations)}. "
            len_mm = f"len(measurement_models)={len(self.measurement_models)}."
            raise ValueError(errormsg + len_obs + len_loc + len_mm)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, item):
        return (
            self.locations[item],
            self.observations[item],
            self.measurement_models[item],
        )


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
        Jacobian of the ODE vector-field :math:`f=f(t,y)` with respect to the :math:`y` variable.
    ddf
        Hessian of the ODE vector-field :math:`f=f(t,y)` with respect to the :math:`y` variable.
    solution
        Closed form, analytic solution to the problem. Used for testing and benchmarking.
    dy0_all
        All initial derivatives up to some order.

    Examples
    --------
    >>> import numpy as np
    >>> def f(t, x):
    ...     return x*(1-x)
    >>> ivp = InitialValueProblem(f, t0=0., tmax=3., y0=0.1)
    >>> ivp.t0, ivp.tmax, ivp.y0
    (0.0, 3.0, 0.1)
    >>> np.round(ivp.f(ivp.t0, ivp.y0), 2)
    0.09
    """

    f: Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float
    y0: Union[FloatArgType, np.ndarray]
    df: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
    ddf: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    @property
    def dimension(self):
        if np.isscalar(self.y0):
            return 1
        return len(self.y0)


@dataclasses.dataclass
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
    >>> linsys = LinearSystem(A, b)
    >>> linsys
    LinearSystem(A=array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), b=array([0, 1, 2]), solution=None)
    """

    A: Union[
        np.ndarray,
        scipy.sparse.spmatrix,
        linops.LinearOperator,
        randvars.RandomVariable,
    ]
    b: Union[np.ndarray, randvars.RandomVariable]

    # For testing and benchmarking
    solution: Optional[Union[np.ndarray, randvars.RandomVariable]] = None


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
        A number or a vector representing the lower bounds of the integrals.
    upper_bd
        A number or a vector representing the upper bounds of the integrals.
    output_dim
        Output dimension of the integrand.
    solution
        Closed form, analytic solution to the problem. Used for testing and benchmarking.

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
    >>>
    >>> lower_bd = [0., 0.]
    >>> upper_bd = [1., 1.]
    >>> qp2d = QuadratureProblem(integrand, lower_bd=lower_bd, upper_bd=upper_bd)
    >>> qp2d.upper_bd
    [1.0, 1.0]
    """

    integrand: Callable[[np.ndarray], Union[float, np.ndarray]]
    lower_bd: Union[FloatArgType, np.ndarray]
    upper_bd: Union[FloatArgType, np.ndarray]
    output_dim: Optional[int] = 1

    # For testing and benchmarking
    solution: Optional[Union[float, np.ndarray, randvars.RandomVariable]] = None
