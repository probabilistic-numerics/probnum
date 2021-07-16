"""Particle filtering and smoothing."""

from ._importance_distributions import (
    BootstrapImportanceDistribution,
    ImportanceDistribution,
    LinearizationImportanceDistribution,
)
from ._particle_filter import ParticleFilter, effective_number_of_events
from ._particle_filter_posterior import ParticleFilterPosterior

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ParticleFilter",
    "ParticleFilterPosterior",
    "effective_number_of_events",
    "ImportanceDistribution",
    "BootstrapImportanceDistribution",
    "LinearizationImportanceDistribution",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ParticleFilter.__module__ = "probnum.filtsmooth.particle"
ParticleFilterPosterior.__module__ = "probnum.filtsmooth.particle"
ImportanceDistribution.__module__ = "probnum.filtsmooth.particle"
BootstrapImportanceDistribution.__module__ = "probnum.filtsmooth.particle"
LinearizationImportanceDistribution.__module__ = "probnum.filtsmooth.particle"
