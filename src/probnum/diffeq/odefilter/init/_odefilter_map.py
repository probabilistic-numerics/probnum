"""MAP estimation for initialization."""

import numpy as np

from probnum import filtsmooth, problems, randprocs, randvars
from probnum.diffeq.odefilter import approx_strategies, information_operators, utils

from ._interface import _InitializationRoutineBase


class ODEFilterMAP(_InitializationRoutineBase):
    def __init__(self, dt=1e-2):
        super().__init__(is_exact=False, requires_jax=False)
        self._dt = dt

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
        gauss_newton = filtsmooth.optim.GaussNewton(kalman=kalman)

        # Compute initial guess and solution
        initial_guess, _ = kalman.filtsmooth(filter_problem)
        solution, _ = gauss_newton.solve(filter_problem, initial_guess=initial_guess)

        # Principally correct, but the ordering is wrong?!?!
        # WHY???
        return solution.states[0]
