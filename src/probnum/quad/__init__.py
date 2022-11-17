"""Quadrature / Numerical Integration of Functions.

This package implements Bayesian quadrature rules used for numerical
integration of functions on a given domain. Such methods integrate a
function by iteratively building a probabilistic model and adaptively
choosing points to evaluate the integrand based on said model.
"""

from probnum.quad._bayesquad import bayesquad, bayesquad_from_data

from . import solvers, kernel_embeddings, integration_measures

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "bayesquad_from_data",
]

# Set correct module paths. Corrects links and module paths in documentation.
bayesquad.__module__ = "probnum.quad"
bayesquad_from_data.__module__ = "probnum.quad"
