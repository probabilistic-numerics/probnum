"""Integration measures for Bayesian quadrature methods."""

from ._integration_measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "IntegrationMeasure",
    "GaussianMeasure",
    "LebesgueMeasure",
]

# Set correct module paths. Corrects links and module paths in documentation.
IntegrationMeasure.__module__ = "probnum.quad.integration_measures"
GaussianMeasure.__module__ = "probnum.quad.integration_measures"
LebesgueMeasure.__module__ = "probnum.quad.integration_measures"
