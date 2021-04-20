import numpy as np

from probnum import diffeq, filtsmooth, problems, randvars, statespace

__all__ = ["car_tracking", "logistic_ode", "ornstein_uhlenbeck", "pendulum"]


def car_tracking(delta_t=0.2, measvar=0.5, t_max=20.0):
    r"""State space model for a simple car-tracking scenario.

    A linear, time-invariant Gaussian state space model for car-tracking, based on
    Example 3.6 in Särkkä, 2013. [1]_
    Let :math:`X = (\dot{x}_1, \dot{x}_2, \ddot{x}_1, \ddot{x}_2)`. Then the state
    space model has the following discretized formulation

    .. math::

        X(t_{n}) &=
        \begin{pmatrix}
          0 & 0 & 1 & 0 \\
          0 & 0 & 0 & 1 \\
          0 & 0 & 0 & 0 \\
          0 & 0 & 0 & 0
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
    delta_t : float
        The step size of the discretized time grid on which the model is considered.
    measvar : float
        The marginal measurement variance.
    t_max : float
        The time limit of the grid.

    Returns
    -------
    dynmod
        The `DiscreteLTIGaussian` dynamics model.
    measmod
        The `DiscreteLTIGaussian` measurement model.
    initrv
        A `Normal` random variable for the Gaussian initial conditions.
    regression_problem
        A `RegressionProblem` with time points and noisy observations.

    References
    ----------
    .. [1] Särkkä, Simo. Bayesian Filtering and Smoothing. Cambridge University Press,
        2013.

    """

    dynamat = np.eye(4) + delta_t * np.diag(np.ones(2), 2)
    dynadiff = (
        np.diag(np.array([delta_t ** 3 / 3, delta_t ** 3 / 3, delta_t, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2, delta_t ** 2 / 2]), 2)
        + np.diag(np.array([delta_t ** 2 / 2, delta_t ** 2 / 2]), -2)
    )
    measmat = np.eye(2, 4)
    measdiff = measvar * np.eye(2)
    mean = np.zeros(4)
    cov = 0.5 * measvar * np.eye(4)

    dynmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=dynamat, shift_vec=np.zeros(4), proc_noise_cov_mat=dynadiff
    )
    measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=measmat,
        shift_vec=np.zeros(2),
        proc_noise_cov_mat=measdiff,
    )
    initrv = randvars.Normal(mean, cov)

    # Generate data
    times = np.arange(0.0, t_max, step=delta_t)
    states, obs = statespace.generate_samples(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )
    return dynmod, measmod, initrv, regression_problem


def ornstein_uhlenbeck(delta_t=0.2, lam=0.21, dynvar=0.5, measvar=0.1, t_max=20.0):
    r"""State space model based on an Ornstein Uhlenbeck process.

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
    delta_t : float
        The step size of the discretized time grid on which the model is considered.
    lam : float
        The drift coefficient :math:`\lambda` in the Ornstein-Uhlenbeck dynamics.
    dynvar : float
        The process noise variance.
    measvar: float
        The measurement noise variance.
    t_max : float
        The time limit of the grid.

    Returns
    -------
    dynmod
        The `LTISDE` dynamics model.
    measmod
        The `DiscreteLTIGaussian` measurement model.
    initrv
        A `Normal` random variable for the Gaussian initial conditions.
    regression_problem
        A `RegressionProblem` with time points and noisy observations.


    References
    ----------
    .. [1] Särkkä, Simo, and Solin, Arno. Applied Stochastic Differential Equations.
        Cambridge University Press, 2019
    """

    drift = -lam * np.eye(1)
    force = np.zeros(1)
    disp = np.sqrt(dynvar) * np.eye(1)
    dynmod = statespace.LTISDE(
        driftmat=drift,
        forcevec=force,
        dispmat=disp,
    )
    measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=np.eye(1),
        shift_vec=np.zeros(1),
        proc_noise_cov_mat=measvar * np.eye(1),
    )
    initrv = randvars.Normal(10 * np.ones(1), np.eye(1))

    # Generate data
    times = np.arange(0.0, t_max, step=delta_t)
    states, obs = statespace.generate_samples(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )
    return dynmod, measmod, initrv, regression_problem


def pendulum(delta_t=0.0075, measvar=0.1024, t_max=4.0):
    r"""State space model for a (noisy) pendulum.

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
    delta_t : float
        The step size of the discretized time grid on which the model is considered.
    measvar: float
        The measurement noise variance.
    t_max : float
        The time limit of the grid.

    Returns
    -------
    dynmod
        The `DiscreteGaussian` dynamics model.
    measmod
        The `DiscreteGaussian` measurement model.
    initrv
        A `Normal` random variable for the Gaussian initial conditions.
    regression_problem
        A `RegressionProblem` with time points and noisy observations.


    References
    ----------
    .. [1] Särkkä, Simo. Bayesian Filtering and Smoothing. Cambridge University Press,
        2013.

    """

    # Graviational constant
    g = 9.81

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

    q = 1.0 * (
        np.diag(np.array([delta_t ** 3 / 3, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2]), 1)
        + np.diag(np.array([delta_t ** 2 / 2]), -1)
    )
    r = measvar * np.eye(1)
    initmean = np.ones(2)
    initcov = measvar * np.eye(2)
    dynmod = statespace.DiscreteGaussian(
        input_dim=2,
        output_dim=2,
        state_trans_fun=f,
        proc_noise_cov_mat_fun=lambda t: q,
        jacob_state_trans_fun=df,
    )
    measmod = statespace.DiscreteGaussian(
        input_dim=2,
        output_dim=1,
        state_trans_fun=h,
        proc_noise_cov_mat_fun=lambda t: r,
        jacob_state_trans_fun=dh,
    )
    initrv = randvars.Normal(initmean, initcov)

    # Generate data
    times = np.arange(0.0, t_max, step=delta_t)
    states, obs = statespace.generate_samples(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )
    return dynmod, measmod, initrv, regression_problem


def logistic_ode(delta_t=0.2, t_max=2.0):
    r"""State space model for a probabilistic ODE solver based on the logistic ODE.

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
    dynmod
        The `DiscreteGaussian` dynamics model.
    measmod
        The `DiscreteGaussian` measurement model.
    initrv
        A `Normal` random variable for the Gaussian initial conditions.
    info
        A dictionary containing

            * ``dt`` the step size of the grid,
            * ``tmax`` the limit of the time grid,
            * ``ode`` the right-hand-side of the logistic ODE.

    See Also
    --------
    :py:class:`probnum.diffeq.GaussianIVPFilter`

    """

    logistic = diffeq.logistic(
        (0, t_max), initrv=randvars.Constant(np.array([0.1])), params=(6, 1)
    )
    dynamod = statespace.IBM(ordint=3, spatialdim=1)
    measmod = filtsmooth.DiscreteEKFComponent.from_ode(
        logistic, dynamod, np.zeros((1, 1)), ek0_or_ek1=1
    )

    initmean = np.array([0.1, 0, 0.0, 0.0])
    initcov = np.diag([0.0, 1.0, 1.0, 1.0])
    initrv = randvars.Normal(initmean, initcov)

    return dynamod, measmod, initrv, {"dt": delta_t, "tmax": t_max, "ode": logistic}
