"""Quadrature / Numerical Integration of Functions.

This package implements Bayesian quadrature rules used for numerical
integration of functions on a given domain. Such methods integrate a
function by iteratively building a probabilistic model and adaptively
choosing points to evaluate the integrand based on said model.
"""

from probnum.quad.solvers.policies import Policy, RandomPolicy
from probnum.quad.solvers.stopping_criteria import (
    BQStoppingCriterion,
    ImmediateStop,
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
)

from ._bayesquad import bayesquad, bayesquad_from_data
from ._integration_measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure
from .kernel_embeddings import KernelEmbedding
from .solvers import (
    BayesianQuadrature,
    BQBeliefUpdate,
    BQInfo,
    BQStandardBeliefUpdate,
    BQState,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "bayesquad_from_data",
    "BayesianQuadrature",
    "IntegrationMeasure",
    "ImmediateStop",
    "KernelEmbedding",
    "GaussianMeasure",
    "LebesgueMeasure",
    "BQStoppingCriterion",
    "IntegralVarianceTolerance",
    "MaxNevals",
    "RandomPolicy",
    "RelativeMeanChange",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad"
BQStoppingCriterion.__module__ = "probnum.quad"
ImmediateStop.__module__ = "probnum.quad"
IntegrationMeasure.__module__ = "probnum.quad"
KernelEmbedding.__module__ = "probnum.quad"
GaussianMeasure.__module__ = "probnum.quad"
LebesgueMeasure.__module__ = "probnum.quad"
