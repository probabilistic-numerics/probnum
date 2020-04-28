"""
(Bayesian) Optimization.

Bayesian optimization is a sequential design strategy for global optimization of a (black-box) function (without
requiring derivatives). The optimizer builds an internal model of the function it is optimizing and chooses evaluation
points based on it.
"""

# Public classes and functions. Order is reflected in documentation.

from .objective import *
from .linesearch import *
from .stoppingcriterion import *
from .bayesopt import *
from .deterministic import *
from .stochastic import *
from .optimizer import *
from .optim import *

__all__ = ["LineSearch", "ConstantLearningRate", "BacktrackingLineSearch",
           "Objective", "Eval",
           "StoppingCriterion", "NormOfGradient", "DiffOfFctValues",
           "Optimizer", "RandomSearch", "SteepestDescent", "GradientDescent",
           "NewtonMethod", "LevenbergMarquardt",
           "minimise_rs", "minimise_gd", "minimise_levmarq", "minimise_newton"]

# Set correct module paths (for superclasses). Corrects links and module paths in documentation.
LineSearch.__module__ = "probnum.optim"
StoppingCriterion.__module__ = "probnum.optim"
Optimizer.__module__ = "probnum.optim"
SteepestDescent.__module__ = "probnum.optim"
