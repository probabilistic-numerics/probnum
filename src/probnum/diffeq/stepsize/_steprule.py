"""Rules for adaptive and constant step-size selection."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from probnum.typing import FloatArgType, IntArgType, ToleranceDiffusionType


class StepRule(ABC):
    """Step-size selection rules for ODE solvers."""

    def __init__(self, firststep: FloatArgType):
        self.firststep = firststep

    @abstractmethod
    def suggest(
        self,
        laststep: FloatArgType,
        scaled_error: FloatArgType,
        localconvrate: Optional[IntArgType] = None,
    ):
        """Suggest a new step h_{n+1} given error estimate e_n at step h_n."""
        raise NotImplementedError

    @abstractmethod
    def is_accepted(self, scaled_error: FloatArgType):
        """Check if the proposed step should be accepted or not.

        Variable "proposedstep" not used yet, but may be important in
        the future, e.g. if we decide that instead of tol_per_step (see
        AdaptiveSteps) we want to be able to control tol_per_unitstep.
        """
        raise NotImplementedError

    @abstractmethod
    def errorest_to_norm(
        self, errorest: ToleranceDiffusionType, reference_state: np.ndarray
    ):
        """Computes the norm of error per tolerance (usually referred to as 'E').

        The norm is usually the current error estimate normalised with
        atol, rtol, and the magnitude of the previous states. If this is
        smaller than 1, the step was small enough.
        """
        raise NotImplementedError


class ConstantSteps(StepRule):
    """Constant step-sizes."""

    def __init__(self, stepsize: FloatArgType):
        self.step = stepsize
        super().__init__(firststep=stepsize)

    def suggest(
        self,
        laststep: FloatArgType,
        scaled_error: FloatArgType,
        localconvrate: Optional[IntArgType] = None,
    ):
        return self.step

    def is_accepted(self, scaled_error: FloatArgType):
        """Always True."""
        return True

    def errorest_to_norm(
        self, errorest: ToleranceDiffusionType, reference_state: np.ndarray
    ):
        pass


# Once we have other controls, e.g. PI control, we can rename this into ProportionalControl.
class AdaptiveSteps(StepRule):
    """Adaptive step-size selection (using proportional control).

    Parameters
    ----------
    firststep : float
        First step to be taken by the ODE solver (which happens in absence of error estimates).
    limitchange : list with 2 elements, optional
        Lower and upper bounds for computed change of step.
    safetyscale : float, optional
        Safety factor for proposal of distributions, 0 << safetyscale < 1
    minstep : float, optional
        Minimum step that is allowed. A runtime error is thrown if the proposed step is smaller. Default is 1e-15.
    maxstep : float, optional
        Maximum step that is allowed. A runtime error is thrown if the proposed step is larger. Default is 1e15.
    """

    def __init__(
        self,
        firststep: FloatArgType,
        atol: ToleranceDiffusionType,
        rtol: ToleranceDiffusionType,
        limitchange: Optional[Tuple[FloatArgType]] = (0.2, 10.0),
        safetyscale: Optional[FloatArgType] = 0.95,
        minstep: Optional[FloatArgType] = 1e-15,
        maxstep: Optional[FloatArgType] = 1e15,
    ):
        self.safetyscale = safetyscale
        self.limitchange = limitchange
        self.minstep = minstep
        self.maxstep = maxstep
        self.atol = atol
        self.rtol = rtol
        super().__init__(firststep=firststep)

    def suggest(
        self,
        laststep: FloatArgType,
        scaled_error: FloatArgType,
        localconvrate: Optional[IntArgType] = None,
    ):
        small, large = self.limitchange

        ratio = 1.0 / scaled_error
        change = self.safetyscale * ratio ** (1.0 / localconvrate)

        # The below code should be doable in a single line?
        if change < small:
            step = small * laststep
        elif large < change:
            step = large * laststep
        else:
            step = change * laststep

        if step < self.minstep:
            raise RuntimeError("Step-size smaller than minimum step-size")
        if step > self.maxstep:
            raise RuntimeError("Step-size larger than maximum step-size")
        return step

    def is_accepted(self, scaled_error: FloatArgType):
        return scaled_error < 1

    def errorest_to_norm(
        self, errorest: ToleranceDiffusionType, reference_state: np.ndarray
    ):
        tolerance = self.atol + self.rtol * reference_state
        ratio = errorest / tolerance
        dim = len(ratio) if ratio.ndim > 0 else 1
        return np.linalg.norm(ratio) / np.sqrt(dim)
