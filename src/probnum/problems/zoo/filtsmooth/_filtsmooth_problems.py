from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from probnum import diffeq, filtsmooth, problems, randprocs, randvars
from probnum.problems.zoo import diffeq as diffeq_zoo
from probnum.typing import FloatLike, IntLike

__all__ = [
    "benes_daum",
    "car_tracking",
    "logistic_ode",
    "ornstein_uhlenbeck",
    "pendulum",
]


def car_tracking(
    rng: np.random.Generator,
    measurement_variance: FloatLike = 0.5,
    process_diffusion: FloatLike = 1.0,
    num_prior_derivatives: IntLike = 1,
    timespan: Tuple[FloatLike, FloatLike] = (0.0, 20.0),
    step: FloatLike = 0.2,
    initrv: Optional[randvars.RandomVariable] = None,
    forward_implementation: str = "classic",
    backward_implementation: str = "classic",
):
    r"""Filtering/smoothing setup for a simple car-tracking scenario.

    A discrete, linear, time-invariant Gaussian state space model for car-tracking,
    based on Example 3.6 in Särkkä, 2013. [1]_
    Let :math:`X = (\dot{x}_1, \dot{x}_2, \ddot{x}_1, \ddot{x}_2)`. Then the state
    space model has the following discretized formulation

    .. math::

        X(t_{n}) &=
        \begin{pmatrix}
          1 & 0 & \Delta t& 0 \\
          0 & 1 & 0 & \Delta t \\
          0 & 0 & 1 & 0 \\
          0 & 0 & 0 & 1
        \end{pmatrix} X(t_{n-1})
        +
        q_n \\
        y_{n} &=
        \begin{pmatrix}
          1 & 0 & 0 & 0 \\
          0 & 1 & 0 & 0 \\
        \end{pmatrix} X(t_{n})
        + r_n

    where :math:`q_n \sim \mathcal{N}(0, Q)` and :math:`r_n \sim \mathcal{N}(0, R)`
    for process noise covariance matrix :math:`Q` and measurement noise covariance
    matrix :math:`R`.

    Parameters
    ----------
    rng
        Random number generator.
    measurement_variance
        Marginal measurement variance.
    process_diffusion
        Diffusion constant for the dynamics.
    num_prior_derivatives
        Order of integration for the dynamics model. Defaults to one, which corresponds
        to a Wiener velocity model.
    timespan
        :math:`t_0` and :math:`t_{\max}` of the time grid.
    step
        Step size of the time grid.
    initrv
        Initial random variable.
    forward_implementation
        Implementation of the forward transitions inside prior and measurement model.
        Optional. Default is `classic`. For improved numerical stability, use `sqrt`.
    backward_implementation
        Implementation of the backward transitions inside prior and measurement model.
        Optional. Default is `classic`. For improved numerical stability, use `sqrt`.

    Returns
    -------
    regression_problem
        ``TimeSeriesRegressionProblem`` object with time points and noisy observations.
    info
        Dictionary containing additional information like the prior process.

    References
    ----------
    .. [1] Särkkä, Simo. Bayesian Filtering and Smoothing. Cambridge University Press,
        2013.

    """
    state_dim = 2
    model_dim = state_dim * (num_prior_derivatives + 1)
    measurement_dim = 2
    dynamics_model = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=num_prior_derivatives,
        wiener_process_dimension=state_dim,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    discrete_dynamics_model = dynamics_model.discretise(dt=step)

    measurement_matrix = np.eye(measurement_dim, model_dim)
    measurement_cov = measurement_variance * np.eye(measurement_dim)
    measurement_model = randprocs.markov.discrete.LTIGaussian(
        transition_matrix=measurement_matrix,
        noise=randvars.Normal(mean=np.zeros(measurement_dim), cov=measurement_cov),
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    if initrv is None:
        initrv = randvars.Normal(
            np.zeros(model_dim),
            measurement_variance * np.eye(model_dim),
            cov_cholesky=np.sqrt(measurement_variance) * np.eye(model_dim),
        )

    # Set up regression problem
    time_grid = np.arange(*timespan, step=step)

    prior_process = randprocs.markov.MarkovSequence(
        transition=discrete_dynamics_model, initrv=initrv, initarg=time_grid[0]
    )

    states, obs = randprocs.markov.utils.generate_artificial_measurements(
        rng=rng,
        prior_process=prior_process,
        measmod=measurement_model,
        times=time_grid,
    )
    regression_problem = problems.TimeSeriesRegressionProblem(
        observations=obs,
        locations=time_grid,
        measurement_models=measurement_model,
        solution=states,
    )

    info = dict(prior_process=prior_process)
    return regression_problem, info


def ornstein_uhlenbeck(
    rng: np.random.Generator,
    measurement_variance: FloatLike = 0.1,
    driftspeed: FloatLike = 0.21,
    process_diffusion: FloatLike = 0.5,
    time_grid: Optional[np.ndarray] = None,
    initrv: Optional[randvars.RandomVariable] = None,
    forward_implementation: str = "classic",
    backward_implementation: str = "classic",
):
    r"""Filtering/smoothing setup based on an Ornstein Uhlenbeck process.

    A linear, time-invariant state space model for the dynamics of a time-invariant
    Ornstein-Uhlenbeck process. See e.g. Example 10.19 in Särkkä et. al, 2019. [1]_
    Here, we formulate a continuous-discrete state space model:

    .. math::

        d x(t) &= \lambda x(t) d t + L d w(t) \\
        y_n &= x(t_n) + r_n

    for a drift constant :math:`\lambda` and a driving Wiener process :math:`w(t)`.
    :math:`r_n \sim \mathcal{N}(0, R)` is Gaussian distributed measurement noise
    with covariance matrix :math:`R`.
    Note that the linear, time-invariant dynamics have an equivalent discretization.

    Parameters
    ----------
    rng
        Random number generator.
    measurement_variance
        Marginal measurement variance.
    driftspeed
        Drift parameter of the Ornstein-Uhlenbeck process.
    process_diffusion
        Diffusion constant for the dynamics
    time_grid
        Time grid for the filtering/smoothing problem.
    initrv
        Initial random variable.
    forward_implementation
        Implementation of the forward transitions inside prior and measurement model.
        Optional. Default is `classic`. For improved numerical stability, use `sqrt`.
    backward_implementation
        Implementation of the backward transitions inside prior and measurement model.
        Optional. Default is `classic`. For improved numerical stability, use `sqrt`.


    Returns
    -------
    regression_problem
        ``TimeSeriesRegressionProblem`` object with time points and noisy observations.
    info
        Dictionary containing additional information like the prior process.


    References
    ----------
    .. [1] Särkkä, Simo, and Solin, Arno. Applied Stochastic Differential Equations.
        Cambridge University Press, 2019
    """

    dynamics_model = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition(
        num_derivatives=0,
        wiener_process_dimension=1,
        driftspeed=driftspeed,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    measurement_model = randprocs.markov.discrete.LTIGaussian(
        transition_matrix=np.eye(1),
        noise=randvars.Normal(mean=np.zeros(1), cov=measurement_variance * np.eye(1)),
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    if initrv is None:
        initrv = randvars.Normal(10.0 * np.ones(1), np.eye(1))

    # Set up regression problem
    if time_grid is None:
        time_grid = np.arange(0.0, 20.0, step=0.2)

    prior_process = randprocs.markov.MarkovProcess(
        transition=dynamics_model, initrv=initrv, initarg=time_grid[0]
    )
    states, obs = randprocs.markov.utils.generate_artificial_measurements(
        rng=rng, prior_process=prior_process, measmod=measurement_model, times=time_grid
    )

    regression_problem = problems.TimeSeriesRegressionProblem(
        observations=obs,
        locations=time_grid,
        measurement_models=measurement_model,
        solution=states,
    )

    info = dict(prior_process=prior_process)
    return regression_problem, info


def pendulum(
    rng: np.random.Generator,
    measurement_variance: FloatLike = 0.1024,
    timespan: Tuple[FloatLike, FloatLike] = (0.0, 4.0),
    step: FloatLike = 0.0075,
    initrv: Optional[randvars.RandomVariable] = None,
    initarg: Optional[float] = None,
):
    r"""Filtering/smoothing setup for a (noisy) pendulum.

    A non-linear, discretized state space model for a pendulum with unknown forces
    acting on the dynamics, modeled as Gaussian noise.
    See e.g. Särkkä, 2013 [1]_ for more details.

    .. math::

        \begin{pmatrix}
          x_1(t_n) \\
          x_2(t_n)
        \end{pmatrix}
        &=
        \begin{pmatrix}
          x_1(t_{n-1}) + x_2(t_{n-1}) \cdot h \\
          x_2(t_{n-1}) - g \sin(x_1(t_{n-1})) \cdot h
        \end{pmatrix}
        +
        q_n \\
        y_n &\sim \sin(x_1(t_n)) + r_n

    for some ``step`` size :math:`h` and Gaussian process noise
    :math:`q_n \sim \mathcal{N}(0, Q)` with

    .. math::
        Q =
        \begin{pmatrix}
          \frac{h^3}{3} & \frac{h^2}{2} \\
          \frac{h^2}{2} & h
        \end{pmatrix}

    :math:`g` denotes the gravitational constant and :math:`r_n \sim \mathcal{N}(0, R)`
    is Gaussian mesurement noise with some covariance :math:`R`.

    Parameters
    ----------
    rng
        Random number generator.
    measurement_variance
        Marginal measurement variance.
    timespan
        :math:`t_0` and :math:`t_{\max}` of the time grid.
    step
        Step size of the time grid.
    initrv
        Initial random variable.
    initarg
        Initial time point of the prior process.
        Optional. Default is the left boundary of timespan.

    Returns
    -------
    regression_problem
        ``TimeSeriesRegressionProblem`` object with time points and noisy observations.
    info
        Dictionary containing additional information like the prior process.


    References
    ----------
    .. [1] Särkkä, Simo. Bayesian Filtering and Smoothing. Cambridge University Press,
        2013.

    """

    # Graviational constant
    g = 9.81

    # Define non-linear dynamics and measurements
    def f(t, x):
        x1, x2 = x
        y1 = x1 + x2 * step
        y2 = x2 - g * np.sin(x1) * step
        return np.array([y1, y2])

    def df(t, x):
        x1, _ = x
        y1 = [1, step]
        y2 = [-g * np.cos(x1) * step, 1]
        return np.array([y1, y2])

    def h(t, x):
        x1, _ = x
        return np.array([np.sin(x1)])

    def dh(t, x):
        x1, _ = x
        return np.array([[np.cos(x1), 0.0]])

    noise_cov = (
        np.diag(np.array([step**3 / 3, step]))
        + np.diag(np.array([step**2 / 2]), 1)
        + np.diag(np.array([step**2 / 2]), -1)
    )

    dynamics_model = randprocs.markov.discrete.NonlinearGaussian(
        input_dim=2,
        output_dim=2,
        transition_fun=f,
        noise_fun=lambda t: randvars.Normal(mean=np.zeros(2), cov=noise_cov),
        transition_fun_jacobian=df,
    )

    measurement_model = randprocs.markov.discrete.NonlinearGaussian(
        input_dim=2,
        output_dim=1,
        transition_fun=h,
        noise_fun=lambda t: randvars.Normal(
            mean=np.zeros(1), cov=measurement_variance * np.eye(1)
        ),
        transition_fun_jacobian=dh,
    )

    if initrv is None:
        initrv = randvars.Normal(np.ones(2), measurement_variance * np.eye(2))

    # Generate data
    time_grid = np.arange(*timespan, step=step)

    if initarg is None:
        initarg = time_grid[0]
    prior_process = randprocs.markov.MarkovSequence(
        transition=dynamics_model, initrv=initrv, initarg=initarg
    )

    states, obs = randprocs.markov.utils.generate_artificial_measurements(
        rng=rng,
        prior_process=prior_process,
        measmod=measurement_model,
        times=time_grid,
    )
    regression_problem = problems.TimeSeriesRegressionProblem(
        observations=obs,
        locations=time_grid,
        measurement_models=measurement_model,
        solution=states,
    )

    info = dict(prior_process=prior_process)
    return regression_problem, info


def benes_daum(
    rng: np.random.Generator,
    measurement_variance: FloatLike = 0.1,
    process_diffusion: FloatLike = 1.0,
    time_grid: Optional[np.ndarray] = None,
    initrv: Optional[randvars.RandomVariable] = None,
):
    r"""Filtering/smoothing setup based on the Beneš SDE.

    A non-linear state space model for the dynamics of a Beneš SDE.
    Here, we formulate a continuous-discrete state space model:

    .. math::

        d x(t) &= \tanh(x(t)) d t + L d w(t) \\
        y_n &= x(t_n) + r_n

    for a driving Wiener process :math:`w(t)` and Gaussian distributed measurement noise
    :math:`r_n \sim \mathcal{N}(0, R)` with measurement noise
    covariance matrix :math:`R`.

    Parameters
    ----------
    rng
        Random number generator.
    measurement_variance
        Marginal measurement variance.
    process_diffusion
        Diffusion constant for the dynamics
    time_grid
        Time grid for the filtering/smoothing problem.
    initrv
        Initial random variable.

    Returns
    -------
    regression_problem
        ``TimeSeriesRegressionProblem`` object with time points and noisy observations.
    info
        Dictionary containing additional information like the prior process.

    Notes
    -----
    In order to generate observations for the returned
    ``TimeSeriesRegressionProblem`` object, the non-linear Beneš SDE
    has to be linearized. Here, a ``ContinuousEKFComponent`` is used,
    which corresponds to a first-order linearization as used
    in the extended Kalman filter.
    """

    def f(t, x):
        return np.tanh(x)

    def df(t, x):
        return 1.0 - np.tanh(x) ** 2

    def l(t, x):
        return process_diffusion * np.ones((1, 1))

    if initrv is None:
        initrv = randvars.Normal(np.zeros(1), 3.0 * np.eye(1))

    dynamics_model = randprocs.markov.continuous.SDE(
        state_dimension=1,
        wiener_process_dimension=1,
        drift_function=f,
        dispersion_function=l,
        drift_jacobian=df,
    )
    measurement_model = randprocs.markov.discrete.LTIGaussian(
        transition_matrix=np.eye(1),
        noise=randvars.Normal(mean=np.zeros(1), cov=measurement_variance * np.eye(1)),
    )

    # Generate data
    if time_grid is None:
        time_grid = np.arange(0.0, 4.0, step=0.2)
    # The non-linear dynamics are linearized according to an EKF in order
    # to generate samples.
    linearized_dynamics_model = filtsmooth.gaussian.approx.ContinuousEKFComponent(
        non_linear_model=dynamics_model
    )

    prior_process = randprocs.markov.MarkovProcess(
        transition=dynamics_model, initrv=initrv, initarg=time_grid[0]
    )
    prior_process_with_linearized_dynamics = randprocs.markov.MarkovProcess(
        transition=linearized_dynamics_model, initrv=initrv, initarg=time_grid[0]
    )

    states, obs = randprocs.markov.utils.generate_artificial_measurements(
        rng=rng,
        prior_process=prior_process_with_linearized_dynamics,
        measmod=measurement_model,
        times=time_grid,
    )
    regression_problem = problems.TimeSeriesRegressionProblem(
        observations=obs,
        locations=time_grid,
        measurement_models=measurement_model,
        solution=states,
    )

    info = dict(prior_process=prior_process)
    return regression_problem, info


def logistic_ode(
    y0: Optional[Union[np.ndarray, FloatLike]] = None,
    timespan: Tuple[FloatLike, FloatLike] = (0.0, 2.0),
    step: FloatLike = 0.1,
    params: Tuple[FloatLike, FloatLike] = (6.0, 1.0),
    initrv: Optional[randvars.RandomVariable] = None,
    evlvar: Optional[Union[np.ndarray, FloatLike]] = None,
    ek0_or_ek1: IntLike = 1,
    exclude_initial_condition: bool = True,
    order: IntLike = 3,
    forward_implementation: str = "classic",
    backward_implementation: str = "classic",
):
    r"""Filtering/smoothing setup for a probabilistic ODE solver for the logistic ODE.

    This state space model assumes an integrated Brownian motion prior on the dynamics
    and constructs the ODE likelihood based on the vector field defining the
    logistic ODE.

    Parameters
    ----------
    y0
        Initial conditions of the Initial Value Problem
    timespan
        Time span of the problem
    params
        Parameters for the logistic ODE
    initrv
        Initial random variable of the probabilistic ODE solver
    evlvar
        See :py:class:`probnum.diffeq.ODEFilter`
    ek0_or_ek1
        See :py:class:`probnum.diffeq.ODEFilter`
    exclude_initial_condition
        Whether the resulting regression problem should exclude
        (i.e. not contain) the initial condition of the ODE.
        Optional. Default is True, which means that the initial condition is omitted.
    order
        Order of integration for the Integrated Brownian Motion prior of the solver.
    forward_implementation
        Implementation of the forward transitions inside prior and measurement model.
        Optional. Default is `classic`. For improved numerical stability, use `sqrt`.
    backward_implementation
        Implementation of the backward transitions inside prior and measurement model.
        Optional. Default is `classic`. For improved numerical stability, use `sqrt`.

    Returns
    -------
    regression_problem
        ``TimeSeriesRegressionProblem`` object with time points and zero-observations.
    info
        Dictionary containing additional information like the prior process.

    See Also
    --------
    :py:class:`probnum.diffeq.ODEFilter`

    """

    if y0 is None:
        y0 = np.array([0.1])
    y0 = np.atleast_1d(y0)

    if evlvar is None:
        evlvar = np.zeros((1, 1))

    t0, tmax = timespan

    # Generate ODE regression problem
    logistic_ivp = diffeq_zoo.logistic(t0=t0, tmax=tmax, y0=y0, params=params)
    time_grid = np.arange(*timespan, step=step)
    ode_residual = diffeq.odefilter.information_operators.ODEResidual(
        num_prior_derivatives=order, ode_dimension=logistic_ivp.dimension
    )
    if ek0_or_ek1 == 0:
        ek = diffeq.odefilter.approx_strategies.EK0()
    else:
        ek = diffeq.odefilter.approx_strategies.EK1()
    regression_problem = diffeq.odefilter.utils.ivp_to_regression_problem(
        ivp=logistic_ivp,
        locations=time_grid,
        ode_information_operator=ode_residual,
        approx_strategy=ek,
        ode_measurement_variance=evlvar,
        exclude_initial_condition=exclude_initial_condition,
    )

    # Generate prior process
    if initrv is None:
        initmean = np.array([0.1, 0, 0.0, 0.0])
        initcov = np.diag([0.0, 1.0, 1.0, 1.0])
        initrv = randvars.Normal(initmean, initcov)
    dynamics_model = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=order,
        wiener_process_dimension=1,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    prior_process = randprocs.markov.MarkovProcess(
        transition=dynamics_model, initrv=initrv, initarg=time_grid[0]
    )

    # Return problems and info
    info = dict(
        ivp=logistic_ivp,
        prior_process=prior_process,
    )
    return regression_problem, info
