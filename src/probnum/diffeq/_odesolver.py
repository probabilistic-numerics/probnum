"""ODE solver interface."""

from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np

from probnum import randvars
from probnum.diffeq import callbacks


class ODESolver(ABC):
    """Interface for ODE solvers in ProbNum."""

    @dataclass
    class State:
        """ODE solver states."""

        # In the near future add the IVP here, too.

        t: float
        rv: randvars.RandomVariable
        error_estimate: Optional[np.ndarray] = None

        # The reference state is used for relative error estimation
        reference_state: Optional[np.ndarray] = None

    def __init__(
        self,
        ivp,
        order,
    ):
        self.ivp = ivp
        self.order = order  # e.g.: RK45 has order=5, IBM(q) has order=q
        self.num_steps = 0

    def solve(self, steprule, stop_at_locations=None, callbacks=None):
        """Solve an IVP.

        Parameters
        ----------
        steprule : :class:`StepRule`
            Step-size selection rule, e.g. constant steps or adaptive steps.
        """
        self.steprule = steprule
        times, rvs = [], []
        for state in self.solution_generator(
            steprule, stop_at_locations=stop_at_locations, callbacks=callbacks
        ):
            times.append(state.t)
            rvs.append(state.rv)

        odesol = self.rvlist_to_odesol(times=times, rvs=rvs)
        return self.postprocess(odesol)

    def solution_generator(self, steprule, stop_at_locations=None, callbacks=None):
        """Generate ODE solver steps."""

        callbacks, time_stopper = self._process_event_inputs(
            callbacks, stop_at_locations
        )

        state = self.initialize()
        yield state

        dt = steprule.firststep
        while state.t < self.ivp.tmax:
            if time_stopper is not None:
                dt = time_stopper.adjust_dt_to_time_stamps(state.t, dt)

            state, dt = self.perform_full_step(state, dt, steprule)

            if callbacks is not None:
                for callback in callbacks:
                    state = callback(state)

            self.num_steps += 1
            yield state

    @staticmethod
    def _process_event_inputs(callbacks, stop_at_locations):
        """Process callbacks and time-stamps into a format suitable for solve()."""

        def promote_callback_type(callbacks):
            return callbacks if isinstance(callbacks, abc.Iterable) else [callbacks]

        if callbacks is not None:
            callbacks = promote_callback_type(callbacks)
        if stop_at_locations is not None:
            time_stopper = _TimeStopper(stop_at_locations)
        else:
            time_stopper = None
        return callbacks, time_stopper

    def perform_full_step(self, state, initial_dt, steprule):
        """Perform a full ODE solver step.

        This includes the acceptance/rejection decision as governed by error estimation
        and steprule.
        """
        dt = initial_dt
        step_is_sufficiently_small = False
        while not step_is_sufficiently_small:
            proposed_state = self.attempt_step(state, dt)

            # Acceptance/Rejection due to the step-rule
            internal_norm = steprule.errorest_to_norm(
                errorest=proposed_state.error_estimate,
                reference_state=proposed_state.reference_state,
            )
            step_is_sufficiently_small = steprule.is_accepted(internal_norm)
            suggested_dt = steprule.suggest(
                dt, internal_norm, localconvrate=self.order + 1
            )

            # Get a new step-size for the next step
            if step_is_sufficiently_small:
                dt = min(suggested_dt, self.ivp.tmax - proposed_state.t)
            else:
                dt = min(suggested_dt, self.ivp.tmax - state.t)

        # This line of code is unnecessary?!
        self.method_callback(state)
        return proposed_state, dt

    @abstractmethod
    def initialize(self):
        """Returns t0 and y0 (for the solver, which might be different to ivp.y0)"""
        raise NotImplementedError

    @abstractmethod
    def attempt_step(self, state, dt):
        """Compute a step from the current state to the next state with increment dt.

        This does not include the acceptance/rejection decision from the step-size
        selection. Therefore, if dt turns out to be too large, the result of
        attempt_step() will be discarded.
        """
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


class _TimeStopper:
    """Make the ODE solver stop at specified time-points."""

    def __init__(self, locations: Iterable):
        self._locations = iter(locations)
        self._next_location = next(self._locations)

    def adjust_dt_to_time_stamps(self, t, dt):
        """Check whether the next time-point is supposed to be stopped at."""

        if t >= self._next_location:
            try:
                self._next_location = next(self._locations)
            except StopIteration:
                self._next_location = np.inf

        if t + dt > self._next_location:
            dt = self._next_location - t
        return dt
