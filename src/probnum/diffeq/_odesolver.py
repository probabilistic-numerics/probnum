"""ODE solver interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from probnum import randvars
from probnum.diffeq import events


class ODESolver(ABC):
    """Interface for ODE solvers in ProbNum."""

    @dataclass
    class State:
        """ODE solver states."""

        t: float
        rv: randvars.RandomVariable
        error_estimate: Optional[np.ndarray] = None

        # reference state for relative error estimation
        reference_state: Optional[np.ndarray] = None

    def __init__(
        self,
        ivp,
        order,
        event_handler: Optional[
            Union[events.EventHandler, List[events.EventHandler]]
        ] = None,
    ):
        self.ivp = ivp
        self.order = order  # e.g.: RK45 has order=5, IBM(q) has order=q
        self.num_steps = 0

        if event_handler is not None:
            if isinstance(event_handler, events.EventHandler):
                event_handler = [event_handler]
            for handle in event_handler:
                self.step = handle(self.step)

    def solve(self, steprule):
        """Solve an IVP.

        Parameters
        ----------
        steprule : :class:`StepRule`
            Step-size selection rule, e.g. constant steps or adaptive steps.
        """
        self.steprule = steprule
        times, rvs = [], []
        for state in self.solution_generator(steprule):
            times.append(state.t)
            rvs.append(state.rv)

        odesol = self.rvlist_to_odesol(times=times, rvs=rvs)
        return self.postprocess(odesol)

    def solution_generator(self, steprule):
        """Generate ODE solver steps."""

        state = self.initialize()

        yield state

        stepsize = steprule.firststep

        while state.t < self.ivp.tmax:
            proposed_state = self.step(state, stepsize)
            internal_norm = steprule.errorest_to_norm(
                errorest=proposed_state.error_estimate,
                reference_state=proposed_state.reference_state,
            )
            if steprule.is_accepted(internal_norm):
                self.num_steps += 1
                self.method_callback(proposed_state)

                state = proposed_state
                yield state

            suggested_stepsize = steprule.suggest(
                stepsize, internal_norm, localconvrate=self.order + 1
            )
            stepsize = min(suggested_stepsize, self.ivp.tmax - state.t)

    @abstractmethod
    def initialize(self):
        """Returns t0 and y0 (for the solver, which might be different to ivp.y0)"""
        raise NotImplementedError

    @abstractmethod
    def step(self, state, dt):
        """Every ODE solver needs a step() method that returns a new random variable and
        an error estimate."""
        raise NotImplementedError

    @abstractmethod
    def rvlist_to_odesol(self, times, rvs):
        """Create an ODESolution object."""
        raise NotImplementedError

    def postprocess(self, odesol):
        """Process the ODESolution object before returning."""
        return odesol

    def method_callback(self, state):
        """Optional callback.

        Can be overwritten. Do this as soon as it is clear that the
        current guess is accepted, but before storing it. No return. For
        example: tune hyperparameters (sigma).
        """
        pass
