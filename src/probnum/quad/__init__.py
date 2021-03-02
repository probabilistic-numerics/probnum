"""Bayesian Quadrature."""

from ._bayesquad import *
from .bq_methods import *

from ._integration_measures import (
    IntegrationMeasure,
    LebesgueMeasure,
    GaussianMeasure,
)

from ._kernel_embeddings import (
    KernelEmbedding,
    KExpQuadMLebesgue,
    KExpQuadMGauss,
)

from .bq_methods import (
    BayesianQuadrature,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "BayesianQuadrature",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad"