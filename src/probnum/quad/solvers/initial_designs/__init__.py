"""Initial designs for Bayesian quadrature."""

from ._initial_design import InitialDesign
from ._latin_design import LatinDesign
from ._mc_design import MCDesign

# Public classes and functions. Order is reflected in documentation.
__all__ = ["InitialDesign", "MCDesign", "LatinDesign"]

# Set correct module paths. Corrects links and module paths in documentation.
InitialDesign.__module__ = "probnum.quad.solvers.initial_designs"
MCDesign.__module__ = "probnum.quad.solvers.initial_designs"
LatinDesign.__module__ = "probnum.quad.solvers.initial_designs"
