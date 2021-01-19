"""Linear system beliefs.

Classes representing probabilistic (prior) beliefs over the quantities
of interest of a linear system such as its solution, the matrix inverse
or spectral information.
"""

from ._linear_system import LinearSystemBelief
from ._noisy_symmetric_normal_linear_system import (
    LinearSystemNoise,
    NoisySymmetricNormalLinearSystemBelief,
)
from ._symmetric_normal_linear_system import SymmetricNormalLinearSystemBelief
from ._weak_mean_correspondence import UncertaintyScales, WeakMeanCorrespondenceBelief

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSystemBelief",
    "SymmetricNormalLinearSystemBelief",
    "UncertaintyScales",
    "WeakMeanCorrespondenceBelief",
    "LinearSystemNoise",
    "NoisySymmetricNormalLinearSystemBelief",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSystemBelief.__module__ = "probnum.linalg.solvers.beliefs"
SymmetricNormalLinearSystemBelief.__module__ = "probnum.linalg.solvers.beliefs"
UncertaintyScales.__module__ = "probnum.linalg.solvers.beliefs"
WeakMeanCorrespondenceBelief.__module__ = "probnum.linalg.solvers.beliefs"
LinearSystemNoise.__module__ = "probnum.linalg.solvers.beliefs"
NoisySymmetricNormalLinearSystemBelief.__module__ = "probnum.linalg.solvers.beliefs"
