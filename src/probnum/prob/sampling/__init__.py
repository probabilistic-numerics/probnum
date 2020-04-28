"""
Sampling methods.

This package implements various techniques to sample from probability distributions.
"""

from probnum.prob.sampling.mcmc import *

# Public classes and functions. Order is reflected in documentation.
__all__ = ["rwmh", "mala", "pmala", "hmc", "phmc", "MetropolisHastings", "RandomWalkMH", "HamiltonianMonteCarlo",
           "PreconditionedHamiltonianMonteCarlo", "MetropolisAdjustedLangevinAlgorithm",
           "PreconditionedMetropolisAdjustedLangevinAlgorithm"]

# Set correct module paths (for superclasses). Corrects links and module paths in documentation.
MetropolisHastings.__module__ = "probnum.prob.sampling"
