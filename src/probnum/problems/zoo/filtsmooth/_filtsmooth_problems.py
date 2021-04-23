from typing import Optional, Tuple

import numpy as np

from probnum import diffeq, filtsmooth, problems, randvars, statespace
from probnum.type import FloatArgType

__all__ = ["car_tracking", "logistic_ode", "ornstein_uhlenbeck", "pendulum"]


def car_tracking(
    measurement_variance: FloatArgType = 0.5,
    process_diffusion: FloatArgType = 1.0,
    times: Optional[np.ndarray] = None,
    initrv: Optional[randvars.RandomVariable] = None,
):
    r"""Filtering/smoothing setup for a simple car-tracking scenario.

    A linear, time-invariant Gaussian state space model for car-tracking, based on
    Example 3.6 in Särkkä, 2013. [1]_
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
        q(t_n) \\
        y_{n} &=
        \begin{pmatrix}
          1 & 0 & 0 & 0 \\
          0 & 1 & 0 & 0 \\
        \end{pmatrix} X(t_{n})
        + r_n

    where :math:`q(t_n) \sim \mathcal{N}(0, Q)` and :math:`r_n \sim \mathcal{N}(0, R)`
    for process noise covariance matrix :math:`Q` and measurement noise covariance
    matrix :math:`R`.

    Parameters
    ----------
    measurement_variance
        Marginal measurement variance.
    process_diffusion
        Diffusion constant for the dynamics.
    times
        Time grid for the filtering/smoothing problem.
    initrv
        Initial random variable.

    Returns
    -------
    regression_problem
        `RegressionProblem` object with time points and noisy observations.
    statespace_components
        Dictionary containing
        - dynamics model
        - measurement model
        - initial random variable

    References
    ----------
    .. [1] Särkkä, Simo. Bayesian Filtering and Smoothing. Cambridge University Press,
        2013.

    """

    dynamics_model = statespace.IBM(ordint=1, spatialdim=2)
    dynamics_model.dispmat *= process_diffusion

    measurement_matrix = np.eye(2, 4)
    measurement_cov = measurement_variance * np.eye(2)
    measurement_model = statespace.DiscreteLTIGaussian(
        state_trans_mat=measurement_matrix,
        shift_vec=np.zeros(2),
        proc_noise_cov_mat=measurement_cov,
    )

    if initrv is None:
        initrv = randvars.Normal(np.zeros(4), 0.5 * measurement_variance * np.eye(4))

    # Set up regression problem
    if times is None:
        times = np.arange(0.0, 20.0, step=0.2)
    states, obs = statespace.generate_samples(
        dynmod=dynamics_model, measmod=measurement_model, initrv=initrv, times=times
    )
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )

    statespace_components = dict(
        dynamics_model=dynamics_model,
        measurement_model=measurement_model,
        initrv=initrv,
    )
    return regression_problem, statespace_components


def ornstein_uhlenbeck(
    measurement_variance: FloatArgType = 0.1,
    driftspeed: FloatArgType = 0.21,
    process_diffusion: FloatArgType = 0.5,
    times: Optional[np.ndarray] = None,
    initrv: Optional[randvars.RandomVariable] = None,
):
    r"""Filtering/smoothing setup based on an Ornstein Uhlenbeck process.

    A linear, time-invariant state space model for the dynamics of a time-invariant
    Ornstein-Uhlenbeck process. See e.g. Example 10.19 in Särkkä et. al, 2019. [1]_
    Here, we formulate a continuous-discrete state space model:

    .. math::

        x(t_n) &= \lambda x(t_{n-1}) + q(t_n) \\
        y_n &= x(t_n) + r_n

    for some scalar :math:`\lambda` and :math:`q(t_n) \sim \mathcal{N}(0, Q)`,
    :math:`r_n \sim \mathcal{N}(0, R)` for process noise covariance matrix :math:`Q`
    and measurement noise covariance matrix :math:`R`.
    Note that the linear, time-invariant dynamics have an equivalent discretization.

    Parameters
    ----------
    measurement_variance
        Marginal measurement variance.
    driftspeed
        Drift parameter of the Ornstein-Uhlenbeck process.
    process_diffusion
        Diffusion constant for the dynamics
    times
        Time grid for the filtering/smoothing problem.
    initrv
        Initial random variable.


    Returns
    -------
    regression_problem
        `RegressionProblem` object with time points and noisy observations.
    statespace_components
        Dictionary containing
        - dynamics model
        - measurement model
        - initial random variable


    References
    ----------
    .. [1] Särkkä, Simo, and Solin, Arno. Applied Stochastic Differential Equations.
        Cambridge University Press, 2019
    """

    dynamics_model = statespace.IOUP(ordint=0, spatialdim=1, driftspeed=driftspeed)
    dynamics_model.dispmat *= process_diffusion

    measurement_model = statespace.DiscreteLTIGaussian(
        state_trans_mat=np.eye(1),
        shift_vec=np.zeros(1),
        proc_noise_cov_mat=measurement_variance * np.eye(1),
    )

    if initrv is None:
        initrv = randvars.Normal(10.0 * np.ones(1), np.eye(1))

    # Set up regression problem
    if times is None:
        times = np.arange(0.0, 20.0, step=0.2)
    states, obs = statespace.generate_samples(
        dynmod=dynamics_model, measmod=measurement_model, initrv=initrv, times=times
    )
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )

    statespace_components = dict(
        dynamics_model=dynamics_model,
        measurement_model=measurement_model,
        initrv=initrv,
    )
    return regression_problem, statespace_components


def pendulum(
    measurement_variance: FloatArgType = 0.1024,
    delta_t: FloatArgType = 0.0075,
    t_max: FloatArgType = 4.0,
    initrv: Optional[randvars.RandomVariable] = None,
):
    r"""Filtering/smoothing setup for a (noisy) pendulum.

    A non-linear state space model for a pendulum with unknown forces acting on the
    dynamics, modeled as Gaussian noise. See e.g. Särkkä, 2013 [1]_ for more details.
    Let :math:`X = (\theta, \dot{\theta})`. Then the pendulum model can be formulated
    as the following continuous-discrete state space model:

    .. math::

        \dot{X}(t) &=
        \begin{pmatrix}
          \dot{\theta}(t) \\
          - g \cdot \sin (\theta(t))
        \end{pmatrix}
        +
        \begin{pmatrix}
          0 \\
          1
        \end{pmatrix} W(t) \\
        y_n &= \sin(\theta(t_n))

    where :math:`W(t)` is a one-dimensional Gaussian white noise process.

    Parameters
    ----------
    measurement_variance
        Marginal measurement variance.
    delta_t
        The step size of the discretized time grid on which the model is considered.
    t_max
        The time limit of the grid.
    initrv
        Initial random variable.

    Returns
    -------
    regression_problem
        `RegressionProblem` object with time points and noisy observations.
    statespace_components
        Dictionary containing
        - dynamics model
        - measurement model
        - initial random variable


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
        y1 = x1 + x2 * delta_t
        y2 = x2 - g * np.sin(x1) * delta_t
        return np.array([y1, y2])

    def df(t, x):
        x1, x2 = x
        y1 = [1, delta_t]
        y2 = [-g * np.cos(x1) * delta_t, 1]
        return np.array([y1, y2])

    def h(t, x):
        x1, x2 = x
        return np.array([np.sin(x1)])

    def dh(t, x):
        x1, x2 = x
        return np.array([[np.cos(x1), 0.0]])

    process_noise_cov = (
        np.diag(np.array([delta_t ** 3 / 3, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2]), 1)
        + np.diag(np.array([delta_t ** 2 / 2]), -1)
    )

    dynamics_model = statespace.DiscreteGaussian(
        input_dim=2,
        output_dim=2,
        state_trans_fun=f,
        proc_noise_cov_mat_fun=lambda t: process_noise_cov,
        jacob_state_trans_fun=df,
    )

    measurement_model = statespace.DiscreteGaussian(
        input_dim=2,
        output_dim=1,
        state_trans_fun=h,
        proc_noise_cov_mat_fun=lambda t: measurement_variance * np.eye(1),
        jacob_state_trans_fun=dh,
    )

    if initrv is None:
        initrv = randvars.Normal(np.ones(2), measurement_variance * np.eye(2))

    # Generate data
    times = np.arange(0.0, t_max, step=delta_t)
    states, obs = statespace.generate_samples(
        dynmod=dynamics_model, measmod=measurement_model, initrv=initrv, times=times
    )
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )
    statespace_components = dict(
        dynamics_model=dynamics_model,
        measurement_model=measurement_model,
        initrv=initrv,
    )
    return regression_problem, statespace_components


def logistic_ode(
    ivp_initrv: Optional[randvars.RandomVariable] = None,
    solver_initrv: Optional[randvars.RandomVariable] = None,
    timespan: Optional[Tuple[float, float]] = None,
    params: Optional[Tuple[float, float]] = None,
):
    r"""Filtering/smoothing setup for a probabilistic ODE solver based on the logistic ODE.

    This state space model puts an integrated Brownian motion prior on the dynamics
    and constructs the ODE likelihood based on the vector field defining the
    logistic ODE.

    Parameters
    ----------
    delta_t : float
        The step size of the discretized time grid on which the model is considered.
    t_max : float
        The time limit of the grid.

    Returns
    -------
    logistic_ivp
        The initial value problem based on the logistic ODE.
    statespace_components
        Dictionary containing
        - dynamics model
        - measurement model
        - initial random variable

    See Also
    --------
    :py:class:`probnum.diffeq.GaussianIVPFilter`

    """

    if timespan is None:
        timespan = (0.0, 2.0)

    if ivp_initrv is None:
        ivp_initrv = randvars.Constant(np.array([0.1]))

    if params is None:
        params = (6.0, 1.0)

    logistic_ivp = diffeq.logistic(timespan=timespan, initrv=ivp_initrv, params=params)
    dynamics_model = statespace.IBM(ordint=3, spatialdim=1)
    measurement_model = filtsmooth.DiscreteEKFComponent.from_ode(
        logistic_ivp, prior=dynamics_model, evlvar=np.zeros((1, 1)), ek0_or_ek1=1
    )

    if solver_initrv is None:
        initmean = np.array([0.1, 0, 0.0, 0.0])
        initcov = np.diag([0.0, 1.0, 1.0, 1.0])
        solver_initrv = randvars.Normal(initmean, initcov)

    statespace_components = dict(
        dynamics_model=dynamics_model,
        measurement_model=measurement_model,
        initrv=solver_initrv,
    )
    return logistic_ivp, statespace_components
