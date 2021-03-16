"""Bayesian Quadrature."""

from ._bayesquad import bayesquad
from ._integration_measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure
from ._kernel_embeddings import (
    _KernelEmbedding,
    _KExpQuadMGauss,
    _KExpQuadMLebesgue,
    get_kernel_embedding,
)
from .bq_methods import BayesianQuadrature, sample_from_measure

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "BayesianQuadrature",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad"
