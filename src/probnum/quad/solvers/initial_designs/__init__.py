"""Initial designs for Bayesian quadrature."""

from ._bmc_design import BMCDesign
from ._initial_design import InitialDesign
from ._latin_design import LatinDesign

# Public classes and functions. Order is reflected in documentation.
__all__ = ["InitialDesign", "BMCDesign", "LatinDesign"]

# Set correct module paths. Corrects links and module paths in documentation.
InitialDesign.__module__ = "probnum.quad.solvers.initial_designs"
BMCDesign.__module__ = "probnum.quad.solvers.initial_designs"
LatinDesign.__module__ = "probnum.quad.solvers.initial_designs"
