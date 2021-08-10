"""ODE-Solver as proposed by Abdulle and Garegnani."""

from typing import Callable

import numpy as np

from probnum import _randomvariablelist
from probnum.diffeq import _odesolver, _odesolver_state
from probnum.diffeq.perturbed import scipy_wrapper
from probnum.diffeq.perturbed.step import (
    _perturbation_functions,
    _perturbedstepsolution,
)
from probnum.typing import FloatArgType


class PerturbedStepSolver(_odesolver.ODESolver):
    """Probabilistic ODE solver based on random perturbation of the step-sizes.

    Perturbs the steps accordingly and projects the solution back to the originally
    proposed time points. Proposed by Abdulle and Garegnani (2020) [1]_.

    Parameters
    ----------
    rng :
        Random number generator.
    solver :
        Currently this has to be a Runge-Kutta method based on SciPy.
    noise-scale :
        Scales the amount of noise that is introduced.
    perturb_function :
        Defines how the stepsize is distributed. This can be either one of
        ``perturb_lognormal()`` or ``perturb_uniform()`` or any other perturbation function with
        the same signature.

    References
    ----------
    .. [1] Abdulle, A. and Garegnani, G.
        Random time step probabilistic methods for uncertainty quantification in chaotic and geometric numerical integration.
        Statistics and Computing. 2020.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        solver: scipy_wrapper.WrappedScipyRungeKutta,
        noise_scale: FloatArgType,
        perturb_function: Callable,
    ):
        def perturb_step(rng, step):
            return perturb_function(
                rng=rng,
                step=step,
                solver_order=solver.order,
                noise_scale=noise_scale,
                size=(),
            )

        self.rng = rng
        self.perturb_step = perturb_step
        self.solver = solver
        self.scales = None
        super().__init__(steprule=solver.steprule, order=solver.order)

    @classmethod
    def construct_with_lognormal_perturbation(
        cls,
        rng: np.random.Generator,
        solver: scipy_wrapper.WrappedScipyRungeKutta,
        noise_scale: FloatArgType,
    ):
        pertfun = _perturbation_functions.perturb_lognormal
        return cls(
            rng=rng,
            solver=solver,
            noise_scale=noise_scale,
            perturb_function=pertfun,
        )

    @classmethod
    def construct_with_uniform_perturbation(
        cls,
        rng: np.random.Generator,
        solver: scipy_wrapper.WrappedScipyRungeKutta,
        noise_scale: FloatArgType,
    ):
        pertfun = _perturbation_functions.perturb_uniform
        return cls(
            rng=rng,
            solver=solver,
            noise_scale=noise_scale,
            perturb_function=pertfun,
        )

    def initialize(self, ivp):
        """Initialise and reset the solver."""
        self.scales = []
        return self.solver.initialize(ivp)

    def attempt_step(self, state: _odesolver_state.ODESolverState, dt: FloatArgType):
        """Perturb the original stopping point.

        Perform one perturbed step and project the solution back to the original
        stopping point.

        Parameters
        ----------
        state
            Current state of the ODE solver.
        dt
            Step-size.

        Returns
        -------
        _odesolver_state.ODESolverState
            New state.
        """
        noisy_step = self.perturb_step(self.rng, dt)
        new_state = self.solver.attempt_step(state, noisy_step)
        scale = noisy_step / dt
        self.scales.append(scale)

        t_new = state.t + dt
        state = _odesolver_state.ODESolverState(
            ivp=state.ivp,
            rv=new_state.rv,
            t=t_new,
            error_estimate=new_state.error_estimate,
            reference_state=new_state.reference_state,
        )
        return state

    def method_callback(self, state):
        """Call dense output after each step and store the interpolants."""
        return self.solver.method_callback(state)

    def rvlist_to_odesol(
        self, times: np.ndarray, rvs: _randomvariablelist._RandomVariableList
    ):
        interpolants = self.solver.interpolants
        probnum_solution = _perturbedstepsolution.PerturbedStepSolution(
            self.scales, times, rvs, interpolants
        )
        return probnum_solution

    def postprocess(self, odesol):
        return self.solver.postprocess(odesol)
