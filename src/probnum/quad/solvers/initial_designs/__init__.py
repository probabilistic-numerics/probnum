"""Initial designs for Bayesian quadrature."""

from ._initial_design import InitialDesign

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "InitialDesign",
]

# Set correct module paths. Corrects links and module paths in documentation.
InitialDesign.__module__ = "probnum.quad.solvers.initial_designs"
