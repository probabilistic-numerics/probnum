"""Convenience functions for Gaussian filtering and smoothing.

References
----------
.. [1] https://arxiv.org/pdf/1610.05261.pdf
.. [2] https://arxiv.org/abs/1807.09737
.. [3] https://arxiv.org/abs/1810.03440
.. [4] https://arxiv.org/pdf/2004.00623.pdf
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from probnum import problems, randprocs
from probnum.diffeq import _utils, odefilter
from probnum.typing import ArrayLike, FloatLike

__all__ = ["probsolve_ivp"]

METHODS = {
    "EK0": odefilter.approx_strategies.EK0,
    "EK1": odefilter.approx_strategies.EK1,
}
"""Implemented methods for the filtering-based ODE solver."""


# This interface function is allowed to have many input arguments.
# Having many input arguments implies having many local arguments,
# so we need to disable both here.
# pylint: disable="too-many-arguments,too-many-locals"
def probsolve_ivp(
    f: Callable,
    t0: FloatLike,
    tmax: FloatLike,
    y0: ArrayLike,
    df: Optional[Callable] = None,
    method: str = "EK0",
    dense_output: bool = True,
    algo_order: int = 2,
    adaptive: bool = True,
    atol: FloatLike = 1e-2,
    rtol: FloatLike = 1e-2,
    step: Optional[FloatLike] = None,
    diffusion_model: str = "dynamic",
    time_stops: Optional[ArrayLike] = None,
):
    r"""Solve an initial value problem with a filtering-based ODE solver.

    Numerically computes a Gauss-Markov process which solves numerically
    the initial value problem (IVP) based on a system of first order
    ordinary differential equations (ODEs)

    .. math:: \dot x(t) = f(t, x(t)), \quad x(t_0) = x_0,
        \quad t \in [t_0, T]

    by regarding it as a (nonlinear) Gaussian filtering (and smoothing)
    problem [3]_. For some configurations it recovers certain multistep
    methods [1]_.
    Convergence rates of filtering [2]_ and smoothing [4]_ are
    comparable to those of methods of Runge-Kutta type.


    This function turns a prior-string into an :class:`ODEPrior`, a
    method-string into a filter/smoother of class :class:`GaussFiltSmooth`,
    creates a :class:`ODEFilter` object and calls the :meth:`solve()` method.
    For advanced usage we recommend to do this process manually which
    enables advanced methods of tuning the algorithm.

    This function supports the methods:
    extended Kalman filtering based on a zero-th order Taylor
    approximation (EKF0),
    extended Kalman filtering (EKF1),
    unscented Kalman filtering (UKF),
    extended Kalman smoothing based on a zero-th order Taylor
    approximation (EKS0),
    extended Kalman smoothing (EKS1), and
    unscented Kalman smoothing (UKS).

    For adaptive step-size selection of ODE filters, we implement the
    scheme proposed by Schober et al. (2019), and further examined
    by Bosch et al (2021), where the local error estimate is derived
    from the local, calibrated uncertainty estimate.

    Arguments
    ---------
    f
        ODE vector field.
    t0
        Initial time point.
    tmax
        Final time point.
    y0
        Initial value.
    df
        Jacobian of the ODE vector field.
    adaptive
        Whether to use adaptive steps or not. Default is `True`.
        If `False`, a `step` needs to be specified.
    atol
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    rtol
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    step
        Step size. If atol and rtol are not specified,
        this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen
        as prescribed by :meth:`propose_firststep`.
        Is required only when `adaptive=False`.
    algo_order
        Order of the algorithm. This amounts to choosing the
        number of derivatives of an integrated Wiener process prior.
        For too high orders, process noise covariance matrices become singular.
        For integrated Wiener processes, this maximum seems to be
        ``num_derivatives=11`` (using standard ``float64``).
        It is possible that higher orders may work for you.
        The type of prior relates to prior assumptions
        about the derivative of the solution.
        The higher the order of the algorithm, the faster the convergence,
        but also, the higher-dimensional
        (and thus the more expensive) the state space.
    method : str, optional
        Which method is to be used. Default is ``EK0`` which is the
        method proposed by Schober et al.. The available
        options are

        ================================================  ==============
         Extended Kalman filtering/smoothing (0th order)  ``'EK0'``
         Extended Kalman filtering/smoothing (1st order)  ``'EK1'``
        ================================================  ==============

        First order extended Kalman filtering and smoothing methods (``EK1``)
        require Jacobians of the RHS-vector field of the IVP.
        That is, the argument ``df`` needs to be specified.
        They are likely to perform better than zeroth order methods in
        terms of (A-)stability and "meaningful uncertainty estimates".
        While we recommend to use correct capitalization for the method string,
        lower-case letters will be capitalized internally.
    dense_output : bool
        Whether we want dense output. Optional. Default is ``True``.
        For the ODE filter, dense output requires smoothing,
        so if ``dense_output`` is False, no smoothing is performed;
        but when it is ``True``, the filter solution is smoothed.
    diffusion_model : str
        Which diffusion model to use.
        The choices are ``'constant'`` and ``'dynamic'``,
        which implement different styles of
        online calibration of the underlying diffusion [5]_.
        Optional. Default is ``'dynamic'``.
    time_stops: np.ndarray
        Time-points through which the solver must step. Optional.
        Default is None.

    Returns
    -------
    solution : ODEFilterSolution
        Solution of the ODE problem.

        Can be evaluated at and sampled from at arbitrary grid points.
        Further, it contains fields:

        t : :obj:`np.ndarray`, shape=(N,)
            Mesh used by the solver to compute the solution.
            It includes the initial time :math:`t_0` but not necessarily the
            final time :math:`T`.
        y : :obj:`list` of :obj:`RandomVariable`, length=N
            Discrete-time solution at times :math:`t_1, ..., t_N`,
            as a list of random variables.
            The means and covariances can be accessed with ``solution.y.mean``
            and ``solution.y.cov``.

    Raises
    ------
    ValueError
        If 'diffusion_model' is not in the list of supported diffusion models.
    ValueError
        If 'method' is not in the list of supported methods.

    See Also
    --------
    ODEFilter :
        Solve IVPs with Gaussian filtering and smoothing
    ODEFilterSolution :
        Solution of ODE problems based on Gaussian filtering and smoothing.

    References
    ----------
    .. [1] Schober, M., Särkkä, S. and Hennig, P..
        A probabilistic model for the numerical solution of initial
        value problems.
        Statistics and Computing, 2019.
    .. [2] Kersting, H., Sullivan, T.J., and Hennig, P..
        Convergence rates of Gaussian ODE filters.
        2019.
    .. [3] Tronarp, F., Kersting, H., Särkkä, S., and Hennig, P..
        Probabilistic solutions to ordinary differential equations as
        non-linear Bayesian filtering: a new perspective.
        Statistics and Computing, 2019.
    .. [4] Tronarp, F., Särkkä, S., and Hennig, P..
        Bayesian ODE solvers: the maximum a posteriori estimate.
        2019.
    .. [5] Bosch, N., and Hennig, P., and Tronarp, F..
        Calibrated Adaptive Probabilistic ODE Solvers.
        2021.


    Examples
    --------
    >>> from probnum.diffeq import probsolve_ivp
    >>> import numpy as np

    Solve a simple logistic ODE with fixed steps.

    >>>
    >>> def f(t, x):
    ...     return 4*x*(1-x)
    >>>
    >>> y0 = np.array([0.15])
    >>> t0, tmax = 0., 1.5
    >>> solution = probsolve_ivp(f, t0, tmax, y0, step=0.1, adaptive=False)
    >>> print(np.round(solution.states.mean, 2))
    [[0.15]
     [0.21]
     [0.28]
     [0.37]
     [0.47]
     [0.57]
     [0.66]
     [0.74]
     [0.81]
     [0.87]
     [0.91]
     [0.94]
     [0.96]
     [0.97]
     [0.98]
     [0.99]]


    Other methods are easily accessible.

    >>> def df(t, x):
    ...     return np.array([4. - 8 * x])
    >>> solution = probsolve_ivp(
    ...     f, t0, tmax, y0, df=df, method="EK1",
    ...     algo_order=2, step=0.1, adaptive=False
    ... )
    >>> print(np.round(solution.states.mean, 2))
    [[0.15]
     [0.21]
     [0.28]
     [0.37]
     [0.47]
     [0.57]
     [0.66]
     [0.74]
     [0.81]
     [0.87]
     [0.91]
     [0.93]
     [0.96]
     [0.97]
     [0.98]
     [0.99]]

    """

    # Create IVP object
    ivp = problems.InitialValueProblem(t0=t0, tmax=tmax, y0=np.asarray(y0), f=f, df=df)
    steprule = _utils.construct_steprule(
        ivp=ivp, adaptive=adaptive, step=step, atol=atol, rtol=rtol
    )

    # Construct diffusion model.
    diffusion_model = diffusion_model.lower()
    if diffusion_model not in ["constant", "dynamic"]:
        raise ValueError("Diffusion model is not supported.")

    if diffusion_model == "constant":
        diffusion = randprocs.markov.continuous.ConstantDiffusion()
    else:
        diffusion = randprocs.markov.continuous.PiecewiseConstantDiffusion(t0=ivp.t0)

    # Create solver
    prior_process = randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0,
        num_derivatives=algo_order,
        wiener_process_dimension=ivp.dimension,
        diffuse=True,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )

    method = method.upper()
    if method not in METHODS:
        raise ValueError("Method is not supported.")
    approx_strategy = METHODS[method]()

    solver = odefilter.ODEFilter(
        steprule=steprule,
        prior_process=prior_process,
        approx_strategy=approx_strategy,
        with_smoothing=dense_output,
        diffusion_model=diffusion,
    )

    return solver.solve(ivp=ivp, stop_at=time_stops)
