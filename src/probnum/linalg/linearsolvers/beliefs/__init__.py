"""Linear system beliefs.

Classes representing probabilistic (prior) beliefs over the quantities
of interest of a linear system such as its solution, the matrix inverse
or spectral information.
"""

from ._linear_system_belief import LinearSystemBelief
from ._noisy_linear_system import NoisyLinearSystemBelief
from ._weak_mean_correspondence import WeakMeanCorrespondenceBelief

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSystemBelief",
    "WeakMeanCorrespondenceBelief",
    "NoisyLinearSystemBelief",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSystemBelief.__module__ = "probnum.linalg.linearsolvers.beliefs"
WeakMeanCorrespondenceBelief.__module__ = "probnum.linalg.linearsolvers.beliefs"
NoisyLinearSystemBelief.__module__ = "probnum.linalg.linearsolvers.beliefs"
