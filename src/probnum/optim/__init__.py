"""
(Bayesian) Optimization.

Bayesian optimization is a sequential design strategy for global optimization of a (black-box) function (without
requiring derivatives). The optimizer builds an internal model of the function it is optimizing and chooses evaluation
points based on it.
"""

from probnum.optim.bayesopt import *

# Public classes and functions. Order is reflected in documentation.
__all__ = []