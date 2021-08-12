"""Discrete-time transitions."""

from ._condition_state import condition_state_on_measurement, condition_state_on_rv
from ._linear_gaussian import LinearGaussian
from ._lti_gaussian import LTIGaussian
from ._nonlinear_gaussian import NonlinearGaussian

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "NonlinearGaussian",
    "LinearGaussian",
    "LTIGaussian",
    "condition_state_on_rv",
    "condition_state_on_measurement",
]

# Set correct module paths. Corrects links and module paths in documentation.
NonlinearGaussian.__module__ = "probnum.randprocs.markov.discrete"
LinearGaussian.__module__ = "probnum.randprocs.markov.discrete"
LTIGaussian.__module__ = "probnum.randprocs.markov.discrete"
