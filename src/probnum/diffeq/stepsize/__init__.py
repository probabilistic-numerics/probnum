"""Step-size selection rules."""

from ._propose_firststep import propose_firststep
from ._steprule import AdaptiveSteps, ConstantSteps, StepRule

__all__ = ["StepRule", "AdaptiveSteps", "ConstantSteps", "propose_firststep"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
StepRule.__module__ = "probnum.diffeq.stepsize"
AdaptiveSteps.__module__ = "probnum.diffeq.stepsize"
ConstantSteps.__module__ = "probnum.diffeq.stepsize"
