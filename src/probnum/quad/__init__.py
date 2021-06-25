"""Bayesian Quadrature."""

from ._bayesquad import bayesquad
from ._integration_measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure
from .bq_methods import BayesianQuadrature
from .kernel_embeddings import KernelEmbedding
from .policies import sample_from_measure

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "BayesianQuadrature",
    "IntegrationMeasure",
    "KernelEmbedding",
    "GaussianMeasure",
    "LebesgueMeasure",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad"
IntegrationMeasure.__module__ = "probnum.quad"
KernelEmbedding.__module__ = "probnum.quad"
GaussianMeasure.__module__ = "probnum.quad"
LebesgueMeasure.__module__ = "probnum.quad"
