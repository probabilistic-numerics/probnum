"""MAP estimation for initialization."""

import numpy as np

from probnum import filtsmooth, problems, randprocs, randvars
from probnum.diffeq.odefilter import approx_strategies, information_operators, utils

from ._interface import InitializationRoutine


class ODEFilterMAP(InitializationRoutine):
    """Initialization via maximum-a-posteriori estimation."""

    def __init__(self, *, dt=1e-2, stopping_criterion=None):
        super().__init__(is_exact=False, requires_jax=False)
        self._dt = dt
        self._stopping_criterion = stopping_criterion

    def __call__(
        self,
        *,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:

        # Transform IVP into filtering problem
        info_op = information_operators.ODEResidual(
            num_prior_derivatives=prior_process.transition.num_derivatives,
            ode_dimension=ivp.dimension,
        )
        num_steps = prior_process.transition.num_derivatives + 1
        locations = np.linspace(
            start=ivp.t0,
            stop=ivp.t0 + num_steps * self._dt,
            num=num_steps,
            endpoint=True,
        )
        filter_problem = utils.ivp_to_regression_problem(
            ivp=ivp,
            locations=locations,
            ode_information_operator=info_op,
            approx_strategy=approx_strategies.EK1(),
            ode_measurement_variance=1e-13,
        )

        # Assemble iterated Kalman smoother
        kalman = filtsmooth.gaussian.Kalman(prior_process=prior_process)
        gauss_newton = filtsmooth.optim.GaussNewton(
            kalman=kalman, stopping_criterion=self._stopping_criterion
        )

        # Compute initial guess and solution
        initial_guess, _ = kalman.filtsmooth(filter_problem)
        solution, _ = gauss_newton.solve(filter_problem, initial_guess=initial_guess)

        return solution.states[0]
